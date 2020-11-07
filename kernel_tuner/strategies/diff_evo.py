""" The differential evolution strategy that optimizes the search through the parameter space """
from __future__ import print_function

from scipy.optimize import differential_evolution

from kernel_tuner.strategies.minimize import get_bounds, _cost_func

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

    tuning_options["scaling"] = False
    # build a bounds array as needed for the optimizer
    bounds = get_bounds(tuning_options.tune_params)

    args = (kernel_options, tuning_options, runner, results)

    # call the differential evolution optimizer
    opt_result = differential_evolution(_cost_func, bounds, args, maxiter=1,
                                        polish=False, strategy=method, disp=tuning_options.verbose)

    if tuning_options.verbose:
        print(opt_result.message)

    return results, runner.dev.get_environment()


