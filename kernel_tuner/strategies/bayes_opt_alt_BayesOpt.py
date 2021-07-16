""" BayesianOptimization package from https://github.com/fmfn/BayesianOptimization """
from __future__ import print_function

from collections import OrderedDict
import numpy as np

try:
    from bayes_opt import BayesianOptimization
    bayes_opt_present = True
except Exception:
    BayesianOptimization = None
    bayes_opt_present = False

from kernel_tuner.strategies import minimize

supported_methods = ["poi", "ei", "ucb"]


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

    # defaults as used by Bayesian Optimization Python package
    acq = tuning_options.strategy_options.get("method", "ucb")
    kappa = tuning_options.strategy_options.get("kappa", 2.576)
    xi = tuning_options.strategy_options.get("xi", 0.0)

    tuning_options["scaling"] = True

    results = []

    # function to pass to the optimizer
    def func(**kwargs):
        args = [kwargs[key] for key in tuning_options.tune_params.keys()]
        return -1.0 * minimize._cost_func(args, kernel_options, tuning_options, runner, results)

    bounds, _, _ = minimize.get_bounds_x0_eps(tuning_options)
    pbounds = OrderedDict(zip(tuning_options.tune_params.keys(), bounds))

    verbose = 0
    if tuning_options.verbose:
        verbose = 2

    # print(np.isnan(init_points).any())

    optimizer = BayesianOptimization(f=func, pbounds=pbounds, verbose=verbose)

    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq=acq, kappa=kappa, xi=xi)

    if tuning_options.verbose:
        print(optimizer.max)

    return results, runner.dev.get_environment()
