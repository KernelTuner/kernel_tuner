"""Bayesian Optimization implementation using BO Torch."""

try:
    import torch
    from botorch.acquisition import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
    from botorch.models import SingleTaskGP
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.optim.optimize import optimize_acqf_discrete, optimize_acqf_discrete_local_search
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.sampling import draw_sobol_samples
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.lazy import PsdSumLazyTensor
    from gpytorch.likelihoods import LikelihoodList
    from gpytorch.mlls import ExactMarginalLogLikelihood
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
    bo = BayesianOptimization(searchspace, runner, tuning_options)
    return bo.run(max_fevals)

class BayesianOptimizationTransfer(BayesianOptimization):
    """Bayesian Optimization class with transfer learning."""

    def __init__(self, searchspace: Searchspace, runner, tuning_options):
        super().__init__(searchspace, runner, tuning_options)

        self.best_rgpe_all = []
        self.best_random_all = []
        self.best_vanilla_nei_all = []
        self.noise_std = 0.05

        # Sample data for each base task
        data_by_task = {}
        for task in range(NUM_BASE_TASKS):
            num_training_points = 20
            # draw points from a sobol sequence
            raw_x = draw_sobol_samples(
                bounds=BOUNDS,
                n=num_training_points,
                q=1,
                seed=task + 5397923,
            ).squeeze(1)
            # get observed values
            f_x = f(raw_x, task_shift(task + 1))
            train_y = f_x + noise_std * torch.randn_like(f_x)
            train_yvar = torch.full_like(train_y, noise_std**2)
            # store training data
            data_by_task[task] = {
                # scale x to [0, 1]
                "train_x": normalize(raw_x, bounds=BOUNDS),
                "train_y": train_y,
                "train_yvar": train_yvar,
            }

        # Fit base model
        base_model_list = []
        for task in range(NUM_BASE_TASKS):
            print(f"Fitting base model {task}")
            model = self.get_fitted_model(
                data_by_task[task]["train_x"],
                data_by_task[task]["train_y"],
                data_by_task[task]["train_yvar"],
            )
            base_model_list.append(model)

    def run_config(self, config: tuple):
        return super().run_config(config)
    
    def evaluate_configs(self, X: Tensor):
        return super().evaluate_configs(X)
    
    def initial_sample(self):
        return super().initial_sample()
    
    def initialize_model(self, state_dict=None, exact=True):
        return super().initialize_model(state_dict, exact)
    
    def get_fitted_model(self, train_X, train_Y, train_Yvar, state_dict=None):
        """Get a single task GP. The model will be fit unless a state_dict with model hyperparameters is provided."""
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        if state_dict is None:
            mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
            fit_gpytorch_mll(mll)
        else:
            model.load_state_dict(state_dict)
        return model
    
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
    
    def get_target_model_loocv_sample_preds(self, train_x, train_y, train_yvar, target_model, num_samples):
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
        train_yvar_cv = torch.stack([train_yvar[~m] for m in masks])
        state_dict = target_model.state_dict()
        # expand to batch size of batch_mode LOOCV model
        state_dict_expanded = {
            name: t.expand(batch_size, *[-1 for _ in range(t.ndim)])
            for name, t in state_dict.items()
        }
        model = self.get_fitted_model(
            train_x_cv, train_y_cv, train_yvar_cv, state_dict=state_dict_expanded
        )
        with torch.no_grad():
            posterior = model.posterior(train_x)
            # Since we have a batch mode gp and model.posterior always returns an output dimension,
            # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
            # the last dimension.
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
            return sampler(posterior).squeeze(-1)
    
    def compute_rank_weights(self, train_x, train_y, base_models, target_model, num_samples):
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
        for task in range(len(base_models)):
            model = base_models[task]
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
            mll, model = self.initialize_model()
            fevals_left = max_fevals - self.initial_sample_size

            # Bayesian optimization loop
            for _ in range(fevals_left):

                target_model = get_fitted_model(train_x, train_y, train_yvar)
                model_list = base_model_list + [target_model]
                rank_weights = compute_rank_weights(
                    train_x,
                    train_y,
                    base_model_list,
                    target_model,
                    NUM_POSTERIOR_SAMPLES,
                )

                # fit a Gaussian Process model
                fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch)
                
                # define the acquisition function
                acqf = LogExpectedImprovement(model=model, best_f=self.train_Y.max(), maximize=True)
                
                # optimize acquisition function to find the next evaluation point
                if max_batch_size < self.searchspace_tensors.size(0):
                    # optimize over a lattice if the space is too large
                    candidate, _ = optimize_acqf_discrete_local_search(
                        acqf,
                        q=1,
                        discrete_choices=self.searchspace_tensors,
                        max_batch_size=max_batch_size,
                        num_restarts=5,
                        raw_samples=1024
                    )
                else:
                    candidate, _ = optimize_acqf_discrete(
                        acqf, 
                        q=1, 
                        choices=self.searchspace_tensors,
                        max_batch_size=max_batch_size
                    )
                    
                    # evaluate the new candidate
                    self.evaluate_configs(candidate)
                    fevals_left -= 1

                # reinitialize the models so they are ready for fitting on next iteration
                if fevals_left > 0:
                    mll, model = self.initialize_model(model.state_dict())
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