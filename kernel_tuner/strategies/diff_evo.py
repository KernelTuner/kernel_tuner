""" The differential evolution strategy that optimizes the search through the parameter space """
from scipy.optimize import differential_evolution

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.minimize import get_bounds, _cost_func, scale_from_params
from kernel_tuner import util

supported_methods = ["best1bin", "best1exp", "rand1exp", "randtobest1exp",
                     "best2exp", "rand2exp", "randtobest1bin", "best2bin", "rand2bin", "rand1bin"]


def tune(runner, kernel_options, device_options, tuning_options):
    """ Find the best performing kernel configuration in the parameter space

    :params runner: A runner from kernel_tuner.runners
    :type runner: kernel_tuner.runner

    :param kernel_options: A dictionary with all options for the kernel.
    :type kernel_options: kernel_tuner.interface.Options

    :param device_options: A dictionary with all options for the device
        on which the kernel should be tuned.
    :type device_options: kernel_tuner.interface.Options

    :param tuning_options: A dictionary with all options regarding the tuning
        process.
    :type tuning_options: kernel_tuner.interface.Options

    :returns: A list of dictionaries for executed kernel configurations and their
        execution times. And a dictionary that contains a information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    results = []

    method = tuning_options.strategy_options.get("method", "best1bin")
    popsize = tuning_options.strategy_options.get("popsize", 20)
    maxiter = tuning_options.strategy_options.get("maxiter", 50)

    tuning_options["scaling"] = False
    # build a bounds array as needed for the optimizer
    bounds = get_bounds(tuning_options.tune_params)

    args = (kernel_options, tuning_options, runner, results)

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


