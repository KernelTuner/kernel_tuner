""" The strategy that uses the basinhopping global optimization method """
from __future__ import print_function

import scipy.optimize

from kernel_tuner.strategies.minimize import _cost_func, get_bounds_x0_eps, setup_method_arguments, setup_method_options

def tune(runner, kernel_options, device_options, tuning_options):
    """ Find the best performing kernel configuration in the parameter space

    :params runner: A runner from kernel_tuner.runners
    :type runner: kernel_tuner.runner

    :param kernel_options: A dictionary with all options for the kernel.
    :type kernel_options: dict

    :param device_options: A dictionary with all options for the device
        on which the kernel should be tuned.
    :type device_options: dict

    :param tuning_options: A dictionary with all options regarding the tuning
        process.
    :type tuning_options: dict

    :returns: A list of dictionaries for executed kernel configurations and their
        execution times. And a dictionary that contains a information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    results = []
    cache = {}

    method = tuning_options.method

    #scale variables in x to make 'eps' relevant for multiple variables
    tuning_options["scaling"] = True

    bounds, x0, eps = get_bounds_x0_eps(tuning_options)

    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)
    kwargs['options'] = options

    args = (kernel_options, tuning_options, runner, results, cache)

    minimizer_kwargs = dict(**kwargs)
    minimizer_kwargs["method"] = method
    minimizer_kwargs["args"] = args

    opt_result = scipy.optimize.basinhopping(_cost_func, x0, stepsize=eps, minimizer_kwargs=minimizer_kwargs, disp=tuning_options.verbose)

    if tuning_options.verbose:
        print(opt_result.message)

    return results, runner.dev.get_environment()
