"""Bayesian Optimization implementation using BO Torch."""

import numpy as np

try:
    import torch
    from botorch import fit_gpytorch_model
    from botorch.acquisition import ExpectedImprovement
    from botorch.models import SingleTaskGP
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
    max_fevals = tuning_options.strategy_options.get("max_fevals", 100)
    bo = BayesianOptimization(searchspace, runner, tuning_options)
    return bo.run(max_fevals)

class BayesianOptimization():
    """Bayesian Optimization class."""

    def __init__(self, searchspace: Searchspace, runner, tuning_options):
        self.initial_sample_taken = False
        self.initial_sample_size = tuning_options.strategy_options.get("popsize", 20)
        self.tuning_options = tuning_options
        self.cost_func = CostFunc(searchspace, tuning_options, runner, scaling=False)

        # set up conversion to tensors
        self.searchspace = searchspace
        self.searchspace_tensors = torch.from_numpy(searchspace.get_list_numpy().astype(float))
        self.train_X = torch.empty_like(self.searchspace_tensors)
        self.train_Y = torch.empty(len(self.train_X))

        # get bounds
        bounds = []
        for v in searchspace.params_values:
            bounds.append([min(v), max(v)])
        bounds = torch.from_numpy(np.array(bounds).transpose())

    def evaluate_configs(self, X: Tensor):
        """Evaluate a tensor of one or multiple configurations."""
        if isinstance(X, Tensor):
            results = []
            if X.dim() == 1:
                results = [[self.cost_func(X)]]
            else:
                results = [[self.cost_func(c)] for c in X]
            return torch.from_numpy(np.array(results))
        else:
            raise NotImplementedError(f"Evaluation has not been implemented for type {type(X)}")
        
    def initial_sample(self):
        """Take an initial sample."""
        sample_indices = torch.from_numpy(self.searchspace.get_random_sample_indices(self.initial_sample_size))
        self.train_X = self.searchspace_tensors.index_select(0, sample_indices)
        self.train_Y = self.evaluate_configs(self.train_X)
        self.initial_sample_taken = True

    def run(self, max_fevals: int):
        """Run the Bayesian Optimization loop for at most `max_fevals`."""
        try:
            if not self.initial_sample_taken:
                self.initial_sample()

            # Bayesian optimization loop
            for _ in range(max_fevals):
                # Fit a Gaussian Process model
                gp = SingleTaskGP(self.train_X, self.train_Y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)
                
                # Define the acquisition function
                ei = ExpectedImprovement(model=gp, best_f=self.train_Y.min(), maximize=False)
                
                # Optimize acquisition function to find the next evaluation point
                candidate, _ = optimize_acqf_discrete(
                    ei, 
                    q=1, 
                    choices=self.searchspace_tensors
                )
                
                # Evaluate the new candidate and update the dataset
                new_y = self.evaluate_configs(candidate)
                self.train_X = torch.cat([self.train_X, candidate])
                self.train_Y = torch.cat([self.train_Y, new_y])
        except util.StopCriterionReached as e:
            if self.tuning_options.verbose:
                print(e)

        return self.cost_func.results 
