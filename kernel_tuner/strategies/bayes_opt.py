""" A simple genetic algorithm for parameter search """
from __future__ import print_function

from collections import OrderedDict
from bayes_opt import BayesianOptimization

from kernel_tuner.strategies import minimize

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


    tuning_options["scaling"] = False

    results = []
    cache = {}

    #function to pass to the optimizer
    def func(**kwargs):
        args = [kwargs[key] for key in tuning_options.tune_params.keys()]
        return -1.0 * minimize._cost_func(args, kernel_options, tuning_options, runner, results, cache)

    bounds = minimize.get_bounds(tuning_options.tune_params)
    pbounds = OrderedDict(zip(tuning_options.tune_params.keys(),bounds))

    optimizer = BayesianOptimization(
        f=func,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=10,
    )

    print(optimizer.max)

    return results, runner.dev.get_environment()
