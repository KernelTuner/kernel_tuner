"""The strategy that uses a minimizer method for searching through the parameter space."""

from kernel_tuner.util import StopCriterionReached
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
    get_options,
    snap_to_nearest_config,
    get_strategy_docstring,
)

supported_methods = ["forest", "gbrt", "gp", "dummy"]

_options = dict(
    method=(f"Local optimization algorithm to use, choose any from {supported_methods}", "gp"),
    options=("Options passed to the skopt method as kwargs.", dict()),
    popsize=("Number of initial samples. If `None`, let skopt choose the initial population", None),
    maxiter=("Maximum number of times to repeat the method until the budget is exhausted.", 1),
)


def tune(searchspace: Searchspace, runner, tuning_options):
    import skopt

    method, skopt_options, popsize, maxiter = get_options(tuning_options.strategy_options, _options)

    # Get maximum number of evaluations
    max_fevals = searchspace.size
    if "max_fevals" in tuning_options:
        max_fevals = min(tuning_options["max_fevals"], max_fevals)

    # Set the maximum number of calls to 100 times the maximum number of evaluations.
    # Not all calls by skopt will result in an evaluation since different calls might
    # map to the same configuration.
    if "n_calls" not in skopt_options:
        skopt_options["n_calls"] = 100 * max_fevals

    # If the initial population size is specified, we select `popsize` samples
    # from the search space. This is more efficient than letting skopt select
    # the samples as it is not aware of restrictions.
    if popsize:
        x0 = searchspace.get_random_sample(min(popsize, max_fevals))
        skopt_options["x0"] = [searchspace.get_param_indices(x) for x in x0]

    opt_result = None
    tune_params_values = list(searchspace.tune_params.values())
    bounds = [(0, len(p) - 1) if len(p) > 1 else [0] for p in tune_params_values]

    cost_func = CostFunc(searchspace, tuning_options, runner)
    objective = lambda x: cost_func(searchspace.get_param_config_from_param_indices(x))
    space_constraint = lambda x: searchspace.is_param_config_valid(searchspace.get_param_config_from_param_indices(x))

    skopt_options["space_constraint"] = space_constraint
    skopt_options["verbose"] = tuning_options.verbose

    try:
        for _ in range(maxiter):
            if method == "dummy":
                opt_result = skopt.dummy_minimize(objective, bounds, **skopt_options)
            elif method == "forest":
                opt_result = skopt.forest_minimize(objective, bounds, **skopt_options)
            elif method == "gp":
                opt_result = skopt.gp_minimize(objective, bounds, **skopt_options)
            elif method == "gbrt":
                opt_result = skopt.gbrt_minimize(objective, bounds, **skopt_options)
            else:
                raise ValueError(f"invalid skopt method: {method}")
    except StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(opt_result)

    return cost_func.results


tune.__doc__ = get_strategy_docstring("skopt minimize", _options)
