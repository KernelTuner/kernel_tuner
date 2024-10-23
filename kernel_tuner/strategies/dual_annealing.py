"""The strategy that uses the dual annealing optimization method."""
import scipy.optimize

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc, setup_method_arguments, setup_method_options

supported_methods = ['COBYLA', 'L-BFGS-B', 'SLSQP', 'CG', 'Powell', 'Nelder-Mead', 'BFGS', 'trust-constr']

_options = dict(method=(f"Local optimization method to use, choose any from {supported_methods}", "Powell"))

def tune(searchspace: Searchspace, runner, tuning_options):

    method = common.get_options(tuning_options.strategy_options, _options)[0]

    #scale variables in x to make 'eps' relevant for multiple variables
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=True)

    bounds, x0, _ = cost_func.get_bounds_x0_eps()

    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)
    kwargs['options'] = options


    minimizer_kwargs = {}
    minimizer_kwargs["method"] = method

    opt_result = None
    try:
        opt_result = scipy.optimize.dual_annealing(cost_func, bounds, minimizer_kwargs=minimizer_kwargs, x0=x0)
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(opt_result.message)

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("Dual Annealing", _options)
