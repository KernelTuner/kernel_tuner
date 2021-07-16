""" SciKitOpt's Bayesian Optimization implementation from https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html """
from __future__ import print_function

from collections import OrderedDict
import numpy as np

try:
    from skopt import gp_minimize
    from kernel_tuner import util
    bayes_opt_present = True
except Exception:
    BayesianOptimization = None
    bayes_opt_present = False

from kernel_tuner.strategies import minimize

supported_methods = ["poi", "ei", "ucb", "gp_hedge"]


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

    if not bayes_opt_present:
        raise ImportError("Error: optional dependency Bayesian Optimization not installed")
    init_points = tuning_options.strategy_options.get("popsize", 20)
    n_iter = tuning_options.strategy_options.get("max_fevals", 100)

    #defaults as used by Scikit Python package
    acq = tuning_options.strategy_options.get("method", "gp_hedge")

    tuning_options["scaling"] = True

    results = []
    counter = []

    #function to pass to the optimizer
    def func(args):
        counter.append(1)
        if len(counter) % 50 == 0:
            print(len(counter), flush=True)
        val = minimize._cost_func(args, kernel_options, tuning_options, runner, results)
        return val

    bounds, _, _ = minimize.get_bounds_x0_eps(tuning_options)
    res = gp_minimize(func, bounds, acq_func=acq, n_calls=n_iter, n_initial_points=init_points, n_jobs=-1)

    if tuning_options.verbose:
        print(res)

    return results, runner.dev.get_environment()
