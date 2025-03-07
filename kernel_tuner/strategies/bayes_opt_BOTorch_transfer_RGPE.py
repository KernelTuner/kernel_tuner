"""Bayesian Optimization implementation using BO Torch and transfer learning with RGPE."""

try:
    import torch
    from botorch.acquisition import qLogNoisyExpectedImprovement
    from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.optim.optimize import optimize_acqf_discrete_local_search
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.lazy import PsdSumLazyTensor
    from gpytorch.likelihoods import LikelihoodList
    from gpytorch.models import GP
    from torch import Tensor
    from torch.nn import ModuleList
    bayes_opt_present = True
except ImportError:
    bayes_opt_present = False


from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.bayes_opt_BOTorch import BayesianOptimization
from kernel_tuner.util import StopCriterionReached

# settings
NUM_BASE_TASKS = 5
N_BATCH = 10
NUM_POSTERIOR_SAMPLES = 256
RANDOM_INITIALIZATION_SIZE = 3
N_TRIALS = 10
MC_SAMPLES = 512
N_RESTART_CANDIDATES = 512
N_RESTARTS = 10
Q_BATCH_SIZE = 1


def tune(searchspace: Searchspace, runner, tuning_options):
    """The entry function for tuning a searchspace using this algorithm."""
    max_fevals = tuning_options.strategy_options.get("max_fevals", 100)
    bo = BayesianOptimizationTransfer(searchspace, runner, tuning_options)
    return bo.run(max_fevals)

class BayesianOptimizationTransfer(BayesianOptimization):
    """Bayesian Optimization class with transfer learning."""

    def __init__(self, searchspace: Searchspace, runner, tuning_options):
        super().__init__(searchspace, runner, tuning_options)

        # set up the data and model for each transfer learning base task
        self.searchspaces_transfer_learning: list[Searchspace] = []
        self.inputs_transfer_learning: list[Tensor] = []
        self.outcomes_transfer_learning: list[Tensor] = []
        self.models_transfer_learning: list = []
        for tl_cache in tuning_options.transfer_learning_caches:
            print(f"Importing transfer learning for {tl_cache['kernel_name']}-{tl_cache['device_name']}")
            # construct the searchspace for this task
            tensor_kwargs = searchspace.tensor_kwargs
            tl_searchspace = Searchspace(None, None, None, from_cache=tl_cache)
            tl_searchspace.initialize_tensorspace(**tensor_kwargs)
            self.searchspaces_transfer_learning.append(tl_searchspace)

            # get the inputs and outcomes for this task
            inputs = []
            outcomes = []
            for c in tl_cache["cache"].values():
                result = c[tuning_options.objective]
                if self.is_valid_result(result):
                    config = tuple(c[p] for p in tl_searchspace.tune_params.keys())
                    inputs.append(tl_searchspace.param_config_to_tensor(config))
                    if not self.maximize:
                        result = -result
                    outcomes.append(result)
            tl_inputs = torch.stack(inputs).to(tl_searchspace.tensor_device)
            tl_outcomes = torch.tensor(outcomes, **tensor_kwargs).unsqueeze(-1)
            assert tl_inputs.shape[0] == tl_outcomes.shape[0]
            self.inputs_transfer_learning.append(tl_inputs)
            self.outcomes_transfer_learning.append(tl_outcomes)

            # fit a model and likelihood for this task
            model, mll = self.get_model_and_likelihood(tl_searchspace, tl_inputs, tl_outcomes)
            mll = self.fit(mll)
            self.models_transfer_learning.append(model)
    
    def roll_col(self, X, shift):
        """Rotate columns to right by shift."""
        return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)
    
    def compute_ranking_loss(self, f_samps, target_y):
        """Compute ranking loss for each sample from the posterior over target points.

        Args:
            f_samps: `n_samples x (n) x n`-dim tensor of samples
            target_y: `n x 1`-dim tensor of targets
        Returns:
            Tensor: `n_samples`-dim tensor containing the ranking loss across each sample
        """
        n = target_y.shape[0]
        if f_samps.ndim == 3:
            # Compute ranking loss for target model
            # take cartesian product of target_y
            cartesian_y = torch.cartesian_prod(
                target_y.squeeze(-1),
                target_y.squeeze(-1),
            ).view(n, n, 2)
            # the diagonal of f_samps are the out-of-sample predictions
            # for each LOO model, compare the out of sample predictions to each in-sample prediction
            rank_loss = (
                (
                    (f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps)
                    ^ (cartesian_y[..., 0] < cartesian_y[..., 1])
                )
                .sum(dim=-1)
                .sum(dim=-1)
            )
        else:
            rank_loss = torch.zeros(
                f_samps.shape[0], dtype=torch.long, device=target_y.device
            )
            y_stack = target_y.squeeze(-1).expand(f_samps.shape)
            for i in range(1, target_y.shape[0]):
                rank_loss += (
                    (self.roll_col(f_samps, i) < f_samps) ^ (self.roll_col(y_stack, i) < y_stack)
                ).sum(dim=-1)
        return rank_loss
    
    def get_target_model_loocv_sample_preds(self, train_x, train_y, train_yvar, target_model, num_samples, no_state=False):
        """Create a batch-mode LOOCV GP and draw a joint sample across all points from the target task.

        Args:
            train_x: `n x d` tensor of training points
            train_y: `n x 1` tensor of training targets
            target_model: fitted target model
            num_samples: number of mc samples to draw

        Return: `num_samples x n x n`-dim tensor of samples, where dim=1 represents the `n` LOO models,
            and dim=2 represents the `n` training points.
        """
        batch_size = len(train_x)
        masks = torch.eye(len(train_x), dtype=torch.uint8, device=self.tensor_device).bool()
        train_x_cv = torch.stack([train_x[~m] for m in masks])
        train_y_cv = torch.stack([train_y[~m] for m in masks])
        train_yvar_cv = torch.stack([train_yvar[~m] for m in masks]) if train_yvar is not None else None

        # use a state dictionary for fast updates
        if no_state:
            state_dict_expanded = None
        else:
            state_dict = target_model.state_dict()

            # expand to batch size of batch_mode LOOCV model
            state_dict_expanded = {
                name: t.expand(batch_size, *[-1 for _ in range(t.ndim)])
                for name, t in state_dict.items()
            }
        
        model, _ = self.get_model_and_likelihood(
            self.searchspace, train_x_cv, train_y_cv, train_yvar_cv, state_dict=state_dict_expanded
        )
        with torch.no_grad():
            posterior = model.posterior(train_x)
            # Since we have a batch mode gp and model.posterior always returns an output dimension,
            # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
            # the last dimension.
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
            return sampler(posterior).squeeze(-1)
    
    def compute_rank_weights(self, train_x, train_y, train_yvar, base_models, target_model, num_samples, no_state=False):
        """Compute ranking weights for each base model and the target model (using LOOCV for the target model).
        
        Note: This implementation does not currently address weight dilution, since we only have a small number of base models.

        Args:
            train_x: `n x d` tensor of training points (for target task)
            train_y: `n` tensor of training targets (for target task)
            base_models: list of base models
            target_model: target model
            num_samples: number of mc samples

        Returns:
            Tensor: `n_t`-dim tensor with the ranking weight for each model
        """
        ranking_losses = []

        # compute ranking loss for each base model
        for model in base_models:
            # compute posterior over training points for target task
            posterior = model.posterior(train_x)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
            base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
            # compute and save ranking loss
            ranking_losses.append(self.compute_ranking_loss(base_f_samps, train_y))

        # compute ranking loss for target model using LOOCV
        # f_samps
        target_f_samps = self.get_target_model_loocv_sample_preds(
            train_x,
            train_y,
            train_yvar,
            target_model,
            num_samples,
            no_state=no_state,
        )
        ranking_losses.append(self.compute_ranking_loss(target_f_samps, train_y))
        ranking_loss_tensor = torch.stack(ranking_losses)
        # compute best model (minimum ranking loss) for each sample
        best_models = torch.argmin(ranking_loss_tensor, dim=0)
        # compute proportion of samples for which each model is best
        rank_weights = (
            best_models.bincount(minlength=len(ranking_losses)).type_as(train_x)
            / num_samples
        )
        return rank_weights
    
    def run(self, max_fevals: int, max_batch_size=2048):
        """Run the Bayesian Optimization loop for at most `max_fevals`."""
        try:
            if not self.initial_sample_taken:
                self.initial_sample()
            model, mll = self.get_model_and_likelihood(self.searchspace, self.train_X, self.train_Y, self.train_Yvar)
            fevals_left = max_fevals - self.initial_sample_size
            first_loop = self.initial_sample_size > 0

            # Bayesian optimization loop
            for _ in range(fevals_left):

                # fit a Gaussian Process model
                fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch)

                # calculate the rank weights
                model_list = self.models_transfer_learning + [model]
                rank_weights = self.compute_rank_weights(
                    self.train_X,
                    self.train_Y,
                    self.train_Yvar,
                    self.models_transfer_learning,
                    model,
                    NUM_POSTERIOR_SAMPLES,
                    no_state=first_loop,
                )

                # create rank model and acquisition function
                rgpe_model = RGPE(model_list, rank_weights)
                # acqf = LogExpectedImprovement(model=rgpe_model, best_f=self.train_Y.max(), maximize=True)
                sampler_qnei = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
                qNEI = qLogNoisyExpectedImprovement(
                    model=rgpe_model,
                    X_baseline=self.train_X,
                    sampler=sampler_qnei,
                    prune_baseline=False,
                )

                # optimize
                candidate, _ = optimize_acqf_discrete_local_search(
                    acq_function=qNEI,
                    discrete_choices=self.searchspace_tensors,
                    q=Q_BATCH_SIZE,
                    num_restarts=N_RESTARTS,
                    raw_samples=N_RESTART_CANDIDATES,
                    max_batch_size=max_batch_size
                )
                    
                # evaluate the new candidate
                self.evaluate_configs(candidate)
                fevals_left -= 1

                # reinitialize the models so they are ready for fitting on next iteration
                if fevals_left > 0:
                    model, mll = self.get_model_and_likelihood(self.searchspace, self.train_X, self.train_Y, self.train_Yvar)
                    first_loop = False
        except StopCriterionReached as e:
            if self.tuning_options.verbose:
                print(e)

        return self.cost_func.results


class RGPE(GP, GPyTorchModel):
    """Rank-weighted GP ensemble.
    
    Note: this class inherits from GPyTorchModel which provides an interface for GPyTorch models in botorch.
    """

    _num_outputs = 1  # metadata for botorch

    def __init__(self, models, weights):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "RGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.weights = weights
        self.to(weights)

    def forward(self, x):
        weighted_means = []
        weighted_covars = []
        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights**2 > 0).nonzero()
        non_zero_weights = self.weights[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            model = self.models[raw_idx]
            posterior = model.posterior(x)
            # unstandardize predictions
            posterior_mean = posterior.mean.squeeze(-1)
            posterior_cov = posterior.mvn.lazy_covariance_matrix
            # apply weight
            weight = non_zero_weights[non_zero_weight_idx]
            weighted_means.append(weight * posterior_mean)
            weighted_covars.append(posterior_cov * weight**2)
        # set mean and covariance to be the rank-weighted sum the means and covariances of the
        # base models and target model
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLazyTensor(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)
