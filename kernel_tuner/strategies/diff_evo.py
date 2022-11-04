""" The differential evolution strategy that optimizes the search through the parameter space """
from collections import OrderedDict

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import (_cost_func, get_bounds)
from scipy.optimize import differential_evolution

supported_methods = ["best1bin", "best1exp", "rand1exp", "randtobest1exp",
                     "best2exp", "rand2exp", "randtobest1bin", "best2bin", "rand2bin", "rand1bin"]

_options = OrderedDict(method=(f"Creation method for new population, any of {supported_methods}", "best1bin"),
                       popsize=("Population size", 20),
                       maxiter=("Number of generations", 50))

def tune(runner, tuning_options):

    results = []

    method, popsize, maxiter = common.get_options(tuning_options.strategy_options, _options)

    tuning_options["scaling"] = False
    # build a bounds array as needed for the optimizer
    bounds = get_bounds(tuning_options.tune_params)

    args = (tuning_options, runner, results)

    # ensure particles start from legal points
    searchspace = Searchspace(tuning_options, runner.dev.max_threads)
    population = list(list(p) for p in searchspace.get_random_sample(popsize))

    # call the differential evolution optimizer
    opt_result = None
    try:
        opt_result = differential_evolution(_cost_func, bounds, args, maxiter=maxiter, popsize=popsize, init=population,
                                        polish=False, strategy=method, disp=tuning_options.verbose)
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(opt_result.message)

    return results, runner.dev.get_environment()


tune.__doc__ = common.get_strategy_docstring("Differential Evolution", _options)
