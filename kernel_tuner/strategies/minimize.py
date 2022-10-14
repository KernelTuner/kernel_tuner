""" The strategy that uses a minimizer method for searching through the parameter space """
import logging
import sys
from collections import OrderedDict
from time import perf_counter

import numpy as np
import scipy.optimize
from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common

supported_methods = ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]

_options = OrderedDict(method=(f"Local optimization algorithm to use, choose any from {supported_methods}", "L-BFGS-B"))

def tune(runner, kernel_options, device_options, tuning_options):

    results = []

    method = common.get_options(tuning_options.strategy_options, _options)[0]

    # scale variables in x to make 'eps' relevant for multiple variables
    tuning_options["scaling"] = True

    bounds, x0, _ = get_bounds_x0_eps(tuning_options, runner.dev.max_threads)
    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)

    args = (kernel_options, tuning_options, runner, results)

    opt_result = None
    try:
        opt_result = scipy.optimize.minimize(_cost_func, x0, args=args, method=method, options=options, **kwargs)
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(opt_result.message)

    return results, runner.dev.get_environment()


tune.__doc__ = common.get_strategy_docstring("Minimize", _options)

def _cost_func(x, kernel_options, tuning_options, runner, results, check_restrictions=True):
    """ Cost function used by minimize """
    start_time = perf_counter()
    last_strategy_time = 1000 * (start_time - runner.last_strategy_start_time)

    # error value to return for numeric optimizers that need a numerical value
    error_value = sys.float_info.max if not tuning_options.objective_higher_is_better else -sys.float_info.max
    logging.debug('_cost_func called')
    logging.debug('x: ' + str(x))

    def return_value(record):
        score = record[tuning_options.objective]
        return_val = score if not isinstance(score, util.ErrorConfig) else error_value
        return return_val if not tuning_options.objective_higher_is_better else -return_val

    # check if max_fevals is reached or time limit is exceeded
    util.check_stop_criterion(tuning_options)

    # snap values in x to nearest actual value for each parameter unscale x if needed
    if tuning_options.snap:
        if tuning_options.scaling:
            params = unscale_and_snap_to_nearest(x, tuning_options.tune_params, tuning_options.eps)
        else:
            params = snap_to_nearest_config(x, tuning_options.tune_params)
    else:
        params = x
    logging.debug('params ' + str(params))

    # we cache snapped values, since those correspond to results for an actual instance of the kernel
    x_int = ",".join([str(i) for i in params])
    if x_int in tuning_options.cache:
        cached_result = tuning_options.cache[x_int]
        cached_result['strategy_time'] = last_strategy_time
        results.append(cached_result)
        if x_int not in tuning_options.unique_results:
            util.print_config(cached_result, tuning_options, runner)
            tuning_options.unique_results[x_int] = cached_result
        # upon returning from this function control will be given back to the strategy, so reset the start time
        runner.last_strategy_start_time = perf_counter()
        return return_value(cached_result)

    # check if this is a legal (non-restricted) parameter instance
    if check_restrictions and tuning_options.restrictions:
        params_dict = OrderedDict(zip(tuning_options.tune_params.keys(), params))
        legal = util.check_restrictions(tuning_options.restrictions, params_dict, tuning_options.verbose)
        if not legal:
            error_result = OrderedDict(zip(tuning_options.tune_params.keys(), params))
            error_result[tuning_options.objective] = util.InvalidConfig()
            tuning_options.cache[x_int] = error_result
            return return_value(error_result)

    # compile and benchmark this instance
    res, _ = runner.run([params], kernel_options, tuning_options)

    # get the actual framework time by estimating based on other times
    total_time = 1000 * (perf_counter() - start_time)
    result = res[0]
    if isinstance(result, dict) and 'compile_time' in result and 'verification_time' in result and 'times' in result:
        compile_time = result['compile_time']
        verification_time = result['verification_time']
        total_kernel_time = sum(result['times']) if 'times' in result.keys() else 0
        # substract the other times from the total time to determine the framework time
        result['framework_time'] = max(total_time - (compile_time + verification_time + total_kernel_time), 0)
    result['strategy_time'] = last_strategy_time

    # append to tuning results
    if res:
        tuning_options.unique_results[x_int] = result
        results.append(result)

    # upon returning from this function control will be given back to the strategy, so reset the start time
    runner.last_strategy_start_time = perf_counter()
    return return_value(result)


def get_bounds_x0_eps(tuning_options, max_threads):
    """compute bounds, x0 (the initial guess), and eps"""
    values = list(tuning_options.tune_params.values())

    if "x0" in tuning_options.strategy_options:
        x0 = tuning_options.strategy_options.x0
    else:
        x0 = None

    if tuning_options.scaling:
        eps = np.amin([1.0 / len(v) for v in values])

        # reducing interval from [0, 1] to [0, eps*len(v)]
        bounds = [(0, eps * len(v)) for v in values]
        if x0:
            # x0 has been supplied by the user, map x0 into [0, eps*len(v)]
            x0 = scale_from_params(x0, tuning_options, eps)
        else:
            # get a valid x0
            searchspace = Searchspace(tuning_options, max_threads)
            pos = list(searchspace.get_random_sample(1)[0])
            x0 = scale_from_params(pos, tuning_options.tune_params, eps)
    else:
        bounds = get_bounds(tuning_options.tune_params)
        if not x0:
            x0 = [(min_v + max_v) / 2.0 for (min_v, max_v) in bounds]
        eps = 1e9
        for v_list in values:
            vals = np.sort(v_list)
            eps = min(eps, np.amin(np.gradient(vals)))

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
        sorted_values = np.sort(values)
        bounds.append((sorted_values[0], sorted_values[-1]))
    return bounds


def setup_method_arguments(method, bounds):
    """ prepare method specific arguments """
    kwargs = {}
    # pass bounds to methods that support it
    if method in ["L-BFGS-B", "TNC", "SLSQP"]:
        kwargs['bounds'] = bounds
    return kwargs


def setup_method_options(method, tuning_options):
    """ prepare method specific options """
    kwargs = {}

    # Note that not all methods iterpret maxiter in the same manner
    if "maxiter" in tuning_options.strategy_options:
        maxiter = tuning_options.strategy_options.maxiter
    else:
        maxiter = 100
    kwargs['maxiter'] = maxiter
    if method in ["Nelder-Mead", "Powell"]:
        kwargs['maxfev'] = maxiter
    elif method == "L-BFGS-B":
        kwargs['maxfun'] = maxiter

    # pass eps to methods that support it
    if method in ["CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP"]:
        kwargs['eps'] = tuning_options.eps
    elif method == "COBYLA":
        kwargs['rhobeg'] = tuning_options.eps

    # not all methods support 'disp' option
    if not method in ['TNC']:
        kwargs['disp'] = tuning_options.verbose

    return kwargs


def snap_to_nearest_config(x, tune_params, resolution=1):
    """helper func that for each param selects the closest actual value"""
    params = []
    for i, k in enumerate(tune_params.keys()):
        values = np.array(tune_params[k])
        idx = np.abs(values - x[i]).argmin()
        params.append(values[idx])
    return params


def unscale_and_snap_to_nearest(x, tune_params, eps):
    """helper func that snaps a scaled variable to the nearest config"""
    x_u = [i for i in x]
    for i, v in enumerate(tune_params.values()):
        # create an evenly spaced linear space to map [0,1]-interval
        # to actual values, giving each value an equal chance
        # pad = 0.5/len(v)  #use when interval is [0,1]
        pad = 0.5 * eps    # use when interval is [0, eps*len(v)]
        linspace = np.linspace(pad, (eps * len(v)) - pad, len(v))

        # snap value to nearest point in space, store index
        idx = np.abs(linspace - x[i]).argmin()

        # safeguard that should not be needed
        idx = min(max(idx, 0), len(v) - 1)

        # use index into array of actual values
        x_u[i] = v[idx]
    return x_u


def scale_from_params(params, tune_params, eps):
    """helper func to do the inverse of the 'unscale' function"""
    x = np.zeros(len(params))
    for i, v in enumerate(tune_params.values()):
        x[i] = 0.5 * eps + v.index(params[i])*eps
    return x
