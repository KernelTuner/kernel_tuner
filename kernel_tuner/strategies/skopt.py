"""The strategy that uses the optimizer from skopt for searching through the parameter space."""

import numpy as np
from kernel_tuner.util import StopCriterionReached
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
    get_options,
    get_strategy_docstring,
)

supported_learners = ["RF", "ET", "GBRT", "DUMMY", "GP"]
supported_acq = ["LCB", "EI", "PI","gp_hedge"]
supported_liars = ["cl_min", "cl_mean", "cl_max"]

_options = dict(
    learner=(f"The leaner to use (supported: {supported_learners})", "RF"),
    acq_func=(f"The acquisition function to use (supported: {supported_acq})", "gp_hedge"),
    lie_strategy=(f"The lie strategy to use when using batches (supported: {supported_liars})", "cl_max"),
    kappa=("The value of kappa", 1.96),
    num_initial=("Number of initial samples. If `None`, let skopt choose the initial population", None),
    batch_size=("The number of points to ask per batch", 1),
    skopt_kwargs=("Additional options passed to the skopt `Optimizer` as kwargs.", dict()),
)


def tune(searchspace: Searchspace, runner, tuning_options):
    learner, acq_func, lie_strategy, kappa, num_initial, batch_size, skopt_kwargs = \
            get_options(tuning_options.strategy_options, _options)

    # Get maximum number of evaluations
    max_fevals = min(tuning_options.get("max_fevals", np.inf), searchspace.size)

    # Const function
    cost_func = CostFunc(searchspace, tuning_options, runner)
    opt_config, opt_result = None, None

    # The dimensions. Parameters with one value become categorical
    from skopt.space.space import Categorical, Integer
    tune_params_values = list(searchspace.tune_params.values())
    bounds = [Integer(0, len(p) - 1) if len(p) > 1 else Categorical([0]) for p in tune_params_values]

    # Space constraint
    space_constraint = lambda x: searchspace.is_param_config_valid(
            searchspace.get_param_config_from_param_indices(x))

    # Create skopt optimizer
    skopt_kwargs = dict(skopt_kwargs)
    skopt_kwargs["base_estimator"] = learner
    skopt_kwargs["acq_func"] = acq_func

    # Only set n_initial_points if not None
    if num_initial is not None:
        skopt_kwargs["n_initial_points"] = num_initial

    # Set kappa is not None
    if kappa is not None:
        skopt_kwargs.setdefault("acq_func_kwargs", {})["kappa"] = kappa

    if tuning_options.verbose:
        print(f"Initialize scikit-optimize Optimizer object: {skopt_kwargs}")

    from skopt import Optimizer as SkOptimizer
    optimizer = SkOptimizer(
            dimensions=bounds,
            space_constraint=space_constraint,
            **skopt_kwargs
    )

    # Ask initial batch of configs
    num_initial = optimizer._n_initial_points
    batch = optimizer.ask(num_initial, lie_strategy)
    Xs, Ys = [], []
    eval_count = 0

    if tuning_options.verbose:
        print(f"Asked optimizer for {num_initial} points: {batch}")

    try:
        while eval_count < max_fevals:
            if not batch:
                optimizer.tell(Xs, Ys)
                batch = optimizer.ask(batch_size, lie_strategy)
                Xs, Ys = [], []

                if tuning_options.verbose:
                    print(f"Asked optimizer for {batch_size} points: {batch}")

            x = batch.pop(0)
            y = cost_func(searchspace.get_param_config_from_param_indices(x))
            eval_count += 1

            Xs.append(x)
            Ys.append(y)

            if opt_result is None or y < opt_result:
                opt_config, opt_result = x, y

    except StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    if opt_result is not None and tuning_options.verbose:
        print(f"Best configuration: {opt_result}")

    return cost_func.results


tune.__doc__ = get_strategy_docstring("skopt minimize", _options)
