"""The Pymoo strategy that uses a minimizer method for searching through the parameter space."""

import pymoo.optimize
import pymoo.core

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
    get_options,
    get_strategy_docstring,
    setup_method_arguments,
    setup_method_options,
)

# TODO: Add the PyMOO algorithms
supported_methods = []

_options = dict(method=(f"Pymoo optimization algorithm to use, choose any from {supported_methods}", ""))

def tune(searchspace: Searchspace, runner, tuning_options):

    # TODO:
    # The idea is to create a Problem, Algorithm, and Termination
    # then use to run `pymoo.optimize.minimize`
    # so I basically need to write some adapter/integration code

    method = get_options(tuning_options.strategy_options, _options)[0]

    # scale variables in x to make 'eps' relevant for multiple variables
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=True)

    bounds, x0, _ = cost_func.get_bounds_x0_eps()
    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)

    # TODO: make a pymoo.core.problem.Problem
    # * use `searchspace`, `runner`, and `cost_func` to define the problem
    # * use etc to define the problem
    problem = None # pymoo.core.problem.Problem()

    # TODO: make a pymoo.core.algorithm.Algorithm
    # * use `method` to select the algorithm
    # * use etc to define the algorithm
    algorithm = None # pymoo.core.algorithm.Algorithm()

    # TODO:
    termination = None # pymoo.core.termination.Termination()

    # TODO: change the rest of the code to work with `Pymoo`

    opt_result = None
    try:
        opt_result = pymoo.optimize.minimize(problem, algorithm, termination)
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(opt_result.message)

    return cost_func.results


tune.__doc__ = get_strategy_docstring("Pymoo minimize", _options)
