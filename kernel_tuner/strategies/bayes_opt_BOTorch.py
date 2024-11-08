"""Bayesian Optimization implementation using BO Torch."""

import numpy as np

try:
    import torch
    from botorch import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.models import MixedSingleTaskGP, SingleTaskGP
    from botorch.optim import optimize_acqf_discrete
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch import Tensor
    bayes_opt_present = True
except ImportError:
    bayes_opt_present = False

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
)


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
        self.initial_sample_size = tuning_options.strategy_options.get("popsize", 20)
        self.tuning_options = tuning_options
        self.cost_func = CostFunc(searchspace, tuning_options, runner, scaling=False, return_invalid=True)

        # set up conversion to tensors
        self.searchspace = searchspace
        self.searchspace_tensors = searchspace.get_tensorspace()
        self.train_X = torch.empty(0)
        self.train_Y = torch.empty(0)

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
                self.train_X = torch.cat([self.train_X, torch.from_numpy(np.array(valid_configs))])
                self.train_Y = torch.cat([self.train_Y, torch.from_numpy(np.array(valid_results))])
        else:
            raise NotImplementedError(f"Evaluation has not been implemented for type {type(X)}")
        
    def initial_sample(self):
        """Take an initial sample."""
        sample_indices = torch.from_numpy(self.searchspace.get_random_sample_indices(self.initial_sample_size))
        sample_configs = self.searchspace_tensors.index_select(0, sample_indices)
        self.evaluate_configs(sample_configs)
        self.initial_sample_taken = True

    def initialize_model(self, state_dict=None):
        """Initialize the model, possibly with a state dict for faster fitting."""
        if len(self.searchspace.tensor_categorical_dimensions) == 0:
            model = SingleTaskGP(self.train_X, self.train_Y)
        else:
            model = MixedSingleTaskGP(self.train_X, self.train_Y, self.searchspace.tensor_categorical_dimensions)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # SumMarginalLogLikelihood
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    def run(self, max_fevals: int):
        """Run the Bayesian Optimization loop for at most `max_fevals`."""
        try:
            if not self.initial_sample_taken:
                self.initial_sample()
                mll, model = self.initialize_model()

            # Bayesian optimization loop
            for _ in range(max_fevals):
                # fit a Gaussian Process model
                fit_gpytorch_mll(mll)
                
                # Define the acquisition function
                ei = ExpectedImprovement(model=model, best_f=self.train_Y.min(), maximize=False)
                
                # Optimize acquisition function to find the next evaluation point
                candidate, _ = optimize_acqf_discrete(
                    ei, 
                    q=1, 
                    choices=self.searchspace_tensors
                )
                
                # evaluate the new candidate
                self.evaluate_configs(candidate)

                # reinitialize the models so they are ready for fitting on next iteration
                mll, model = self.initialize_model(model.state_dict())
        except util.StopCriterionReached as e:
            if self.tuning_options.verbose:
                print(e)

        return self.cost_func.results 
