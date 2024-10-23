"""The strategy that uses the basinhopping global optimization method."""
import scipy.optimize

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc, setup_method_arguments, setup_method_options

supported_methods = ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]

_options = dict(method=(f"Local optimization algorithm to use, choose any from {supported_methods}", "L-BFGS-B"),
                       T=("Temperature parameter for the accept or reject criterion", 1.0))

def tune(searchspace: Searchspace, runner, tuning_options):
    method, T = common.get_options(tuning_options.strategy_options, _options)

    # scale variables in x to make 'eps' relevant for multiple variables
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=True)

    bounds, x0, eps = cost_func.get_bounds_x0_eps()

    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)
    kwargs['options'] = options


    minimizer_kwargs = dict(**kwargs)
    minimizer_kwargs["method"] = method

    opt_result = None
    try:
        opt_result = scipy.optimize.basinhopping(cost_func, x0, T=T, stepsize=eps,
                                             minimizer_kwargs=minimizer_kwargs, disp=tuning_options.verbose)
    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(opt_result.message)

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("basin hopping", _options)
