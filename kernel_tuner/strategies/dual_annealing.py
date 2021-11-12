""" The strategy that uses the dual annealing optimization method """
from __future__ import print_function

import scipy.optimize

from kernel_tuner.strategies.minimize import _cost_func, get_bounds_x0_eps, setup_method_arguments, setup_method_options

supported_methods = ['COBYLA','L-BFGS-B','SLSQP','CG','Powell','Nelder-Mead', 'BFGS', 'trust-constr']


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

    method = tuning_options.strategy_options.get("method", "Powell")

    #scale variables in x to make 'eps' relevant for multiple variables
    tuning_options["scaling"] = True

    bounds, _, _ = get_bounds_x0_eps(tuning_options)

    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)
    kwargs['options'] = options

    args = (kernel_options, tuning_options, runner, results)

    minimizer_kwargs = dict()
    minimizer_kwargs["method"] = method

    opt_result = scipy.optimize.dual_annealing(_cost_func, bounds,
            args= args, local_search_options=minimizer_kwargs)

    if tuning_options.verbose:
        print(opt_result.message)

    return results, runner.dev.get_environment()
