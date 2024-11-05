"""Bayesian Optimization implementation using BO Torch."""

import numpy as np
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
)


def tune(searchspace: Searchspace, runner, tuning_options):
    max_fevals = tuning_options.strategy_options.get("max_fevals", 100)
    initial_sample_size = tuning_options.strategy_options.get("popsize", 20)
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=False)

    # function to optimize
    def evaluate_function(X):
        if isinstance(X, (Tensor, list)):
            results = []
            if X.dim() == 1:
                results = [[cost_func(X)]]
            else:
                results = [[cost_func(c)] for c in X]
            return torch.from_numpy(np.array(results))
        else:
            raise NotImplementedError(f"Evaluation has not been implemented for type {type(X)}")

    # set up conversion to tensors
    full_space = torch.from_numpy(searchspace.get_list_numpy().astype(float))

    # get bounds
    bounds = []
    for v in searchspace.params_values:
        bounds.append([min(v), max(v)])
    bounds = torch.from_numpy(np.array(bounds).transpose())

    try:
        # take initial sample
        sample_indices = torch.from_numpy(searchspace.get_random_sample_indices(initial_sample_size))
        train_X = full_space.index_select(0, sample_indices)
        train_Y = evaluate_function(train_X)

        # Bayesian optimization loop
        for _ in range(max_fevals):
            # Fit a Gaussian Process model
            gp = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)
            
            # Define the acquisition function
            ei = ExpectedImprovement(model=gp, best_f=train_Y.min(), maximize=False)
            
            # Optimize acquisition function to find the next evaluation point
            candidate, _ = optimize_acqf_discrete(
                ei, 
                q=1, 
                choices=full_space
            )
            
            # Evaluate the new candidate and update the dataset
            new_y = evaluate_function(candidate)
            train_X = torch.cat([train_X, candidate])
            train_Y = torch.cat([train_Y, new_y])
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    return cost_func.results
