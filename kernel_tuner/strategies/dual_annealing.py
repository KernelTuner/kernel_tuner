""" The strategy that uses the dual annealing optimization method """
from collections import OrderedDict

import scipy.optimize
from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import (_cost_func, get_bounds_x0_eps,
                                            setup_method_arguments,
                                            setup_method_options)

supported_methods = ['COBYLA', 'L-BFGS-B', 'SLSQP', 'CG', 'Powell', 'Nelder-Mead', 'BFGS', 'trust-constr']

_options = OrderedDict(method=(f"Local optimization method to use, choose any from {supported_methods}", "Powell"))

def tune(searchspace: Searchspace, runner, tuning_options):

    results = []

    method = common.get_options(tuning_options.strategy_options, _options)[0]

    #scale variables in x to make 'eps' relevant for multiple variables
    tuning_options["scaling"] = True

    bounds, x0, _ = get_bounds_x0_eps(searchspace, tuning_options)

    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)
    kwargs['options'] = options

    args = (tuning_options, runner, results)

    minimizer_kwargs = {}
    minimizer_kwargs["method"] = method

    opt_result = None
    try:
        opt_result = scipy.optimize.dual_annealing(_cost_func, bounds, args=args, minimizer_kwargs=minimizer_kwargs, x0=x0)
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(opt_result.message)

    return results


tune.__doc__ = common.get_strategy_docstring("Dual Annealing", _options)
