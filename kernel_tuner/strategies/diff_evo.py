""" The differential evolution strategy that optimizes the search through the parameter space """
from __future__ import print_function

import numpy
from scipy.optimize import differential_evolution
from kernel_tuner import util

import kernel_tuner.strategies.minimize as minimize

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

    #build a bounds array as needed for the optimizer
    bounds = minimize.get_bounds(tuning_options.tune_params)

    #call the differential evolution optimizer
    opt_result = differential_evolution(_cost_func, bounds, [kernel_options, tuning_options, runner, results],
                                        maxiter=1, polish=False, disp=tuning_options.verbose)

    if tuning_options.verbose:
        print(opt_result.message)
        print('best config:', minimize.snap_to_nearest_config(opt_result.x, tuning_options.tune_params))

    return results, runner.dev.get_environment()



def _cost_func(x, kernel_options, tuning_options, runner, results):
    """ Cost function used by the differential evolution optimizer """

    #snap values in x to nearest actual value for each parameter
    params = minimize.snap_to_nearest_config(x, tuning_options.tune_params)

    #check if this is a legal (non-restricted) parameter instance
    if tuning_options.restrictions:
        legal = util.check_restrictions(tuning_options.restrictions, params, tuning_options.tune_params.keys(), tuning_options.verbose)
        if not legal:
            return 1e20

    #compile and benchmark this instance
    res, _ = runner.run([params], kernel_options, tuning_options)

    #append to tuning results
    if len(res) > 0:
        results.append(res[0])
        return res[0]['time']

    return 1e20
