""" A simple genetic algorithm for parameter search """
from __future__ import print_function

from collections import OrderedDict
from bayes_opt import BayesianOptimization

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

    #Bayesian Optimization strategy seems to need some hyper parameter tuning to
    #become better than random sampling for auto-tuning GPU kernels.

    #alpha, normalize_y, and n_restarts_optimizer are options to
    #https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    #defaults used by Baysian Optimization are:
    #   alpha=1e-6,  #1e-3 recommended for very noisy or discrete search spaces
    #   n_restarts_optimizer=5,
    #   normalize_y=True,

    #several exploration friendly settings are: (default is acq="ucb", kappa=2.576)
    #   acq="poi", xi=1e-1
    #   acq="ei", xi=1e-1
    #   acq="ucb", kappa=10

    #defaults as used by Bayesian Optimization Python package
    acq = tuning_options.strategy_options.get("method", "poi")
    kappa = tuning_options.strategy_options.get("kappa", 2.576)
    xi = tuning_options.strategy_options.get("xi", 0.0)
    init_points = tuning_options.strategy_options.get("popsize", 5)
    n_iter = tuning_options.strategy_options.get("maxiter", 25)
    alpha = tuning_options.strategy_options.get("alpha", 1e-6)

    tuning_options["scaling"] = True

    results = []

    #function to pass to the optimizer
    def func(**kwargs):
        args = [kwargs[key] for key in tuning_options.tune_params.keys()]
        return -1.0 * minimize._cost_func(args, kernel_options, tuning_options, runner, results)

    bounds, _, _ = minimize.get_bounds_x0_eps(tuning_options)
    pbounds = OrderedDict(zip(tuning_options.tune_params.keys(),bounds))

    verbose=0
    if tuning_options.verbose:
        verbose=2

    optimizer = BayesianOptimization(f=func, pbounds=pbounds, verbose=verbose, alpha=alpha)

    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq=acq, kappa=kappa, xi=xi)

    if tuning_options.verbose:
        print(optimizer.max)

    return results, runner.dev.get_environment()
