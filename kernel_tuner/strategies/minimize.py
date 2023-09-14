"""The strategy that uses a minimizer method for searching through the parameter space."""

import scipy.optimize

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
    get_options,
    get_strategy_docstring,
    setup_method_arguments,
    setup_method_options,
)

supported_methods = ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]

_options = dict(method=(f"Local optimization algorithm to use, choose any from {supported_methods}", "L-BFGS-B"))

def tune(searchspace: Searchspace, runner, tuning_options):

    method = get_options(tuning_options.strategy_options, _options)[0]

    # scale variables in x to make 'eps' relevant for multiple variables
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=True)

    bounds, x0, _ = cost_func.get_bounds_x0_eps()
    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)

    opt_result = None
    try:
        opt_result = scipy.optimize.minimize(cost_func, x0, method=method, options=options, **kwargs)
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(opt_result.message)

    return cost_func.results


tune.__doc__ = get_strategy_docstring("Minimize", _options)
