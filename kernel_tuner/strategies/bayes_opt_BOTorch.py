"""Bayesian Optimization implementation using BO Torch."""

from math import ceil

import numpy as np

try:
    import torch
    from botorch import fit_gpytorch_mll
    from botorch.acquisition import (
        LogExpectedImprovement,
        ProbabilityOfImprovement,
        qExpectedUtilityOfBestOption,
        qLogExpectedImprovement,
        qLowerBoundMaxValueEntropy,
    )
    from botorch.models import MixedSingleTaskGP, SingleTaskGP, SingleTaskVariationalGP
    from botorch.models.transforms import Normalize, Standardize
    from botorch.optim import optimize_acqf_discrete, optimize_acqf_discrete_local_search
    from botorch.optim.fit import fit_gpytorch_mll_torch
    from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
    from torch import Tensor
    bayes_opt_present = True
except ImportError:
    bayes_opt_present = False

import gpytorch.settings as gp_settings
import linear_operator.settings as linop_settings

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
)

# set gpytorch to approximate mode for faster fitting
linop_settings._fast_covar_root_decomposition._default = True
linop_settings._fast_log_prob._default = True
linop_settings._fast_solves._default = True
linop_settings.cholesky_max_tries._global_value = 6
linop_settings.max_cholesky_size._global_value = 800
gp_settings.max_eager_kernel_size._global_value = 800


def tune(searchspace: Searchspace, runner, tuning_options):
    """The entry function for tuning a searchspace using this algorithm."""
    max_fevals = tuning_options.strategy_options.get("max_fevals", 100)
    bo = BayesianOptimization(searchspace, runner, tuning_options)
    return bo.run(max_fevals)

class BayesianOptimization():
    """Bayesian Optimization class."""

    def __init__(self, searchspace: Searchspace, runner, tuning_options):
        """Initialization of the Bayesian Optimization class. Does not evaluate configurations."""
        self.initial_sample_taken = False
        self.initial_sample_size: int = tuning_options.strategy_options.get("popsize", 20)
        self.tuning_options = tuning_options
        self.cost_func = CostFunc(searchspace, tuning_options, runner, scaling=False, return_invalid=True)

        # select the device to use (CUDA or Apple Silicon MPS if available)
        # TODO keep an eye on Apple Silicon support. Currently `linalg_cholesky` is not yet implemented for MPS.
        self.tensor_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set up conversion to tensors
        self.searchspace = searchspace
        self.searchspace.initialize_tensorspace(dtype=torch.float32, device=self.tensor_device)
        self.searchspace_tensors = searchspace.get_tensorspace()
        self.bounds, self.bounds_indices = self.searchspace.get_tensorspace_bounds()
        self.train_X = torch.empty(0, **self.searchspace.tensor_kwargs)
        self.train_Y = torch.empty(0, **self.searchspace.tensor_kwargs)

    def run_config(self, config: tuple):
        """Run a single configuration. Returns the result and whether it is valid."""
        result = self.cost_func(config)
        valid = not isinstance(result, util.ErrorConfig) and not np.isnan(result)
        if not valid:
            result = np.nan
        return [result], valid

    def evaluate_configs(self, X: Tensor):
        """Evaluate a tensor of one or multiple configurations. Modifies train_X and train_Y accordingly."""
        if isinstance(X, Tensor):
            valid_configs = []
            valid_results = []
            if X.dim() == 1:
                X = [X]
            for config in X:
                assert isinstance(config, Tensor), f"Config must be a Tensor, but is of type {type(config)} ({config})"
                param_config = self.searchspace.tensor_to_param_config(config)
                res, valid = self.run_config(param_config)
                if valid:
                    valid_configs.append(config)
                    valid_results.append(res)
                
                # remove evaluated configurations from the full searchspace
                index = self.searchspace.get_param_config_index(param_config)
                self.searchspace_tensors = torch.cat((self.searchspace_tensors[:index], 
                                                      self.searchspace_tensors[index+1:]))

            # add valid results to the training set
            if len(valid_configs) > 0 and len(valid_results) > 0:
                self.train_X = torch.cat([self.train_X, torch.stack(valid_configs)])
                self.train_Y = torch.cat([self.train_Y, torch.tensor(valid_results, **self.searchspace.tensor_kwargs)])
        else:
            raise NotImplementedError(f"Evaluation has not been implemented for type {type(X)}")
        
    def initial_sample(self):
        """Take an initial sample."""
        sample_indices = torch.from_numpy(self.searchspace.get_random_sample_indices(self.initial_sample_size)).to(self.tensor_device)
        sample_configs = self.searchspace_tensors.index_select(0, sample_indices)
        self.evaluate_configs(sample_configs)
        self.initial_sample_taken = True

    def initialize_model(self, state_dict=None, exact=True):
        """Initialize the model and likelihood, possibly with a state dict for faster fitting."""
        train_X = self.train_X
        train_Y = self.train_Y
        # transforms = dict(input_transform=Normalize(train_X.dim()), outcome_transform=Standardize(train_Y.dim()))
        transforms = dict(input_transform=Normalize(d=train_X.shape[-1], indices=self.bounds_indices, bounds=self.bounds))

        # initialize the model
        if exact:
            catdims = self.searchspace.get_tensorspace_categorical_dimensions()
            if len(catdims) == 0:
                model = SingleTaskGP(train_X, train_Y, **transforms)
            else:
                model = MixedSingleTaskGP(train_X, train_Y, cat_dims=catdims, **transforms)
        else:
            model = SingleTaskVariationalGP(train_X, train_Y, **transforms)

        # load the previous state
        if exact and state_dict is not None:
            model.load_state_dict(state_dict)

        # initialize the likelihood
        if exact:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        else:
            mll = VariationalELBO(model.likelihood, model.model, num_data=train_Y.size(0))
        return mll, model

    def run(self, max_fevals: int, feval_per_loop=10, max_batch_size=2048):
        """Run the Bayesian Optimization loop for at most `max_fevals`."""
        try:
            if not self.initial_sample_taken:
                self.initial_sample()
            mll, model = self.initialize_model()
            num_fevals = self.initial_sample_size

            # Bayesian optimization loop
            max_loops = ceil(max_fevals/feval_per_loop)
            for f in range(max_loops):
                # fit a Gaussian Process model
                fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch)
                
                # define the acquisition function
                acqf = LogExpectedImprovement(model=model, best_f=self.train_Y.min(), maximize=False)
                # acqf = NoisyExpectedImprovement(model=model, , maximize=False)
                # acqf = ProbabilityOfImprovement(model=model, best_f=self.train_Y.min(), maximize=False)
                # acqf = qLowerBoundMaxValueEntropy(model=model, candidate_set=self.searchspace_tensors, maximize=False)
                # acqf = qLogExpectedImprovement(model=model, best_f=self.train_Y.min())
                # acqf = qExpectedUtilityOfBestOption(pref_model=model)

                # divide the optimization space into random chuncks
                tensorspace_size = self.searchspace_tensors.size(0)
                num_optimization_spaces = max(min(feval_per_loop, max_fevals-num_fevals), ceil(tensorspace_size / max_batch_size))
                if num_optimization_spaces <= 1:
                    optimization_spaces = [self.searchspace_tensors]
                else:
                    # shuffle the searchspace
                    shuffled_indices = torch.randperm(tensorspace_size)
                    tensorspace = self.searchspace_tensors[shuffled_indices]
                    optimization_spaces = tensorspace.split(ceil(tensorspace_size / num_optimization_spaces))
                
                # optimize acquisition function to find the next evaluation point
                for optimization_space in optimization_spaces:

                    # optimize over a lattice if the space is too large
                    if max_batch_size < optimization_space.size(0):
                        candidate, _ = optimize_acqf_discrete_local_search(
                            acqf, 
                            q=1,
                            discrete_choices=optimization_space, 
                            max_batch_size=max_batch_size,
                            num_restarts=5,
                            raw_samples=1024
                        )
                    else:
                        candidate, _ = optimize_acqf_discrete(
                            acqf, 
                            q=1, 
                            choices=optimization_space,
                            max_batch_size=max_batch_size
                        )
                    
                    # evaluate the new candidate
                    self.evaluate_configs(candidate)
                    num_fevals += 1

                # reinitialize the models so they are ready for fitting on next iteration
                if f < max_loops - 1:
                    mll, model = self.initialize_model(model.state_dict())
        except util.StopCriterionReached as e:
            if self.tuning_options.verbose:
                print(e)

        return self.cost_func.results 
