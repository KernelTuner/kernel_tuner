""" The strategy that uses a minimizer method for searching through the parameter space """
import logging
import sys
from collections import OrderedDict
from time import perf_counter

import numpy as np
import scipy.optimize
from kernel_tuner import util
from kernel_tuner.strategies.common import (_cost_func, get_bounds_x0_eps,
                                            get_options,
                                            get_strategy_docstring,
                                            setup_method_arguments,
                                            setup_method_options)

supported_methods = ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]

_options = OrderedDict(method=(f"Local optimization algorithm to use, choose any from {supported_methods}", "L-BFGS-B"))

def tune(runner, kernel_options, device_options, tuning_options):

    results = []

    method = get_options(tuning_options.strategy_options, _options)[0]

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


tune.__doc__ = get_strategy_docstring("Minimize", _options)
