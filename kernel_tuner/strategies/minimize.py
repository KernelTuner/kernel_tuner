""" The strategy that uses a minimizer method for searching through the parameter space """
from __future__ import print_function

import logging

import numpy
import scipy.optimize
from kernel_tuner import util

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
    cache = {}

    method = tuning_options.method

    #scale variables in x to make 'eps' relevant for multiple variables
    tuning_options["scaling"] = True

    bounds, x0, _ = get_bounds_x0_eps(tuning_options)
    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)

    #not all methods support 'disp' option
    if not method in ['TNC']:
        options['disp'] = tuning_options.verbose

    args = (kernel_options, tuning_options, runner, results, cache)

    opt_result = scipy.optimize.minimize(_cost_func, x0, args=args, method=method, options=options, **kwargs)

    if tuning_options.verbose:
        print(opt_result.message)

    return results, runner.dev.get_environment()


def _cost_func(x, kernel_options, tuning_options, runner, results, cache):
    """ Cost function used by minimize """

    error_time = 1e20
    logging.debug('_cost_func called')
    logging.debug('x: ' + str(x))

    x_key = ",".join([str(i) for i in x])
    if x_key in cache:
        return cache[x_key]

    #snap values in x to nearest actual value for each parameter unscale x if needed
    if tuning_options.scaling:
        params = unscale_and_snap_to_nearest(x, tuning_options.tune_params, tuning_options.eps)
    else:
        params = snap_to_nearest_config(x, tuning_options.tune_params)

    logging.debug('params ' + str(params))

    x_int = ",".join([str(i) for i in params])
    if x_int in cache:
        return cache[x_int]

    #check if this is a legal (non-restricted) parameter instance
    if tuning_options.restrictions:
        legal = util.check_restrictions(tuning_options.restrictions, params, tuning_options.tune_params.keys(), tuning_options.verbose)
        if not legal:
            cache[x_int] = error_time
            cache[x_key] = error_time
            return error_time

    #compile and benchmark this instance
    res, _ = runner.run([params], kernel_options, tuning_options)

    #append to tuning results
    if res:
        results.append(res[0])
        cache[x_int] = res[0]['time']
        cache[x_key] = res[0]['time']
        return res[0]['time']

    cache[x_int] = error_time
    cache[x_key] = error_time
    return error_time



def get_bounds_x0_eps(tuning_options):
    """compute bounds, x0 (the initial guess), and eps"""
    values = tuning_options.tune_params.values()

    if tuning_options.scaling:
        #bounds = [(0, 1) for _ in values]
        #x0 = [0.5 for _ in bounds]
        eps = numpy.amin([1.0/len(v) for v in values])

        #reducing interval from [0, 1] to [0, eps*len(v)]
        bounds = [(0, eps*len(v)) for v in values]
        x0 = [0.5*eps*len(v) for v in values]
    else:
        bounds = get_bounds(tuning_options.tune_params)
        x0 = [(min_v+max_v)/2.0 for (min_v, max_v) in bounds]
        eps = 1e9
        for v_list in values:
            vals = numpy.sort(v_list)
            eps = min(eps, numpy.amin(numpy.gradient(vals)))

    tuning_options["eps"] = eps
    logging.debug('get_bounds_x0_eps called')
    logging.debug('bounds ' + str(bounds))
    logging.debug('x0 ' + str(x0))
    logging.debug('eps ' + str(eps))

    return bounds, x0, eps


def get_bounds(tune_params):
    """ create a bounds array from the tunable parameters """
    bounds = []
    for values in tune_params.values():
        sorted_values = numpy.sort(values)
        bounds.append((sorted_values[0], sorted_values[-1]))
    return bounds


def setup_method_arguments(method, bounds):
    """ prepare method specific arguments """
    kwargs = {}
    #pass bounds to methods that support it
    if method in ["L-BFGS-B", "TNC", "SLSQP"]:
        kwargs['bounds'] = bounds
    return kwargs


def setup_method_options(method, tuning_options):
    """ prepare method specific options """
    kwargs = {}

    #pass size of parameter space as max iterations to methods that support it
    #it seems not all methods iterpret this value in the same manner
    maxiter = numpy.prod([len(v) for v in tuning_options.tune_params.values()])
    kwargs['maxiter'] = maxiter
    if method in ["Nelder-Mead", "Powell"]:
        kwargs['maxfev'] = maxiter
    elif method == "L-BFGS-B":
        kwargs['maxfun'] = maxiter

    #pass eps to methods that support it
    if method in ["CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP"]:
        kwargs['eps'] = tuning_options.eps
    elif method == "COBYLA":
        kwargs['rhobeg'] = tuning_options.eps

    return kwargs


def snap_to_nearest_config(x, tune_params):
    """helper func that for each param selects the closest actual value"""
    params = []
    for i, k in enumerate(tune_params.keys()):
        values = numpy.array(tune_params[k])
        idx = numpy.abs(values-x[i]).argmin()
        params.append(int(values[idx]))
    return params


def unscale_and_snap_to_nearest(x, tune_params, eps):
    """helper func that snaps a scaled variable to the nearest config"""
    x_u = [i for i in x]
    for i, v in enumerate(tune_params.values()):
        #create an evenly spaced linear space to map [0,1]-interval
        #to actual values, giving each value an equal chance
        #pad = 0.5/len(v)  #use when interval is [0,1]
        pad = 0.5*eps      #use when interval is [0, eps*len(v)]
        linspace = numpy.linspace(pad, (eps*len(v))-pad, len(v))

        #snap value to nearest point in space, store index
        idx = numpy.abs(linspace-x[i]).argmin()

        #safeguard that should not be needed
        idx = min(max(idx, 0), len(v)-1)

        #use index into array of actual values
        x_u[i] = v[idx]
    return x_u
