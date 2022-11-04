import logging
import sys
from collections import OrderedDict
from time import perf_counter

import numpy as np
from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace

_docstring_template = """ Find the best performing kernel configuration in the parameter space

    This $NAME$ strategy supports the following strategy_options:

$STRAT_OPT$

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
        execution times. And a dictionary that contains information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """


def get_strategy_docstring(name, strategy_options):
    """ Generate docstring for a 'tune' method of a strategy """
    return _docstring_template.replace("$NAME$", name).replace("$STRAT_OPT$", make_strategy_options_doc(strategy_options))


def make_strategy_options_doc(strategy_options):
    """ Generate documentation for the supported strategy options and their defaults """
    doc = ""
    for opt, val in strategy_options.items():
        doc += f"     * {opt}: {val[0]}, default {str(val[1])}. \n"
    doc += "\n"
    return doc


def get_options(strategy_options, options):
    """ Get the strategy-specific options or their defaults from user-supplied strategy_options """
    accepted = list(options.keys()) + ["max_fevals", "time_limit"]
    for key in strategy_options:
        if key not in accepted:
            raise ValueError(f"Unrecognized option {key} in strategy_options")
    assert isinstance(options, OrderedDict)
    return [strategy_options.get(opt, default) for opt, (_, default) in options.items()]


def _cost_func(x, kernel_options, tuning_options, runner, results, check_restrictions=True):
    """ Cost function used by almost all strategies """
    runner.last_strategy_time = 1000 * (perf_counter() - runner.last_strategy_start_time)

    # error value to return for numeric optimizers that need a numerical value
    logging.debug('_cost_func called')
    logging.debug('x: ' + str(x))

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

    legal = True
    result = {}
    x_int = ",".join([str(i) for i in params])

    # else check if this is a legal (non-restricted) configuration
    if check_restrictions and tuning_options.restrictions:
        params_dict = OrderedDict(zip(tuning_options.tune_params.keys(), params))
        legal = util.check_restrictions(tuning_options.restrictions, params_dict, tuning_options.verbose)
        if not legal:
            result = params_dict
            result[tuning_options.objective] = util.InvalidConfig()

    # compile and benchmark this instance
    if not result:
        res, _ = runner.run([params], kernel_options, tuning_options)
        result = res[0]

    # append to tuning results
    if x_int not in tuning_options.unique_results:
        tuning_options.unique_results[x_int] = result

    results.append(result)

    # get numerical return value, taking optimization direction into account
    return_value = result[tuning_options.objective] or sys.float_info.max
    return_value = return_value if not tuning_options.objective_higher_is_better else -return_value

    # upon returning from this function control will be given back to the strategy, so reset the start time
    runner.last_strategy_start_time = perf_counter()
    return return_value


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
    if method not in ['TNC']:
        kwargs['disp'] = tuning_options.verbose

    return kwargs


def snap_to_nearest_config(x, tune_params):
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
        # use when interval is [0, eps*len(v)]
        pad = 0.5 * eps
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
