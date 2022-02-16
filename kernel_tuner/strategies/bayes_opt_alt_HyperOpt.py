""" HyperOpt package from https://github.com/hyperopt/hyperopt """
from __future__ import print_function

from collections import OrderedDict
import numpy as np

try:
    from hyperopt import hp, fmin, tpe, space_eval, base, STATUS_FAIL
    import itertools
    from kernel_tuner import util
    bayes_opt_present = True
except Exception:
    hp = None
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
    n_iter = tuning_options.strategy_options.get("max_fevals", 100)
    tuning_options["scaling"] = True
    results = []

    #function to pass to the optimizer
    def func(params):
        print(params)
        param_config = list(params.values())
        print(param_config)
        if not util.config_valid(param_config, tuning_options, runner.dev.max_threads):
            return {
                'status': STATUS_FAIL
            }
        return minimize._cost_func(param_config, kernel_options, tuning_options, runner, results)

    minimize.get_bounds_x0_eps(tuning_options)    # necessary to have EPS set
    tune_params = tuning_options.tune_params
    space = dict()
    for tune_param in tune_params.keys():
        space[tune_param] = hp.choice(tune_param, tune_params[tune_param])

    trials = base.Trials()
    fmin(func, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)

    return results, runner.dev.get_environment()
