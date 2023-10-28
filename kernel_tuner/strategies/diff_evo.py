"""The differential evolution strategy that optimizes the search through the parameter space."""
from scipy.optimize import differential_evolution

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc

supported_methods = ["best1bin", "best1exp", "rand1exp", "randtobest1exp", "best2exp", "rand2exp", "randtobest1bin", "best2bin", "rand2bin", "rand1bin"]

_options = dict(method=(f"Creation method for new population, any of {supported_methods}", "best1bin"),
                       popsize=("Population size", 20),
                       maxiter=("Number of generations", 100))


def tune(searchspace: Searchspace, runner, tuning_options):


    method, popsize, maxiter = common.get_options(tuning_options.strategy_options, _options)

    # build a bounds array as needed for the optimizer
    cost_func = CostFunc(searchspace, tuning_options, runner)
    bounds = cost_func.get_bounds()

    # ensure particles start from legal points
    population = list(list(p) for p in searchspace.get_random_sample(popsize))

    # call the differential evolution optimizer
    opt_result = None
    try:
        opt_result = differential_evolution(cost_func, bounds, maxiter=maxiter, popsize=popsize, init=population,
                                        polish=False, strategy=method, disp=tuning_options.verbose)
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(opt_result.message)

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("Differential Evolution", _options)
