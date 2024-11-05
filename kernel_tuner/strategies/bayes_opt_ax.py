"""Bayesian Optimization implementation using the Ax platform."""

from ax import optimize

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
)


def tune(searchspace: Searchspace, runner, tuning_options):
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=True)

    ax_searchspace = searchspace.to_ax_searchspace()

    try:
        best_parameters, best_values, experiment, model = optimize(
            parameters=ax_searchspace.parameters,
            parameter_constraints=ax_searchspace.parameter_constraints,
            # Booth function
            evaluation_function=cost_func,
            minimize=True,
        )
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    return cost_func.results
