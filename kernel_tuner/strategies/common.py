"""Module for functionality that is commonly used throughout the strategies."""

import logging
import sys
from time import perf_counter

import numpy as np
from scipy.spatial import distance

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace

_docstring_template = """ Find the best performing kernel configuration in the parameter space

    This $NAME$ strategy supports the following strategy_options:

$STRAT_OPT$

    :params runner: A runner from kernel_tuner.runners
    :type runner: kernel_tuner.runner

    :param tuning_options: A dictionary with all options regarding the tuning
        process.
    :type tuning_options: kernel_tuner.interface.Options

    :returns: A list of dictionaries for executed kernel configurations and their
        execution times. And a dictionary that contains information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """


def get_strategy_docstring(name, strategy_options):
    """Generate docstring for a 'tune' method of a strategy."""
    return _docstring_template.replace("$NAME$", name).replace(
        "$STRAT_OPT$", make_strategy_options_doc(strategy_options)
    )


def make_strategy_options_doc(strategy_options):
    """Generate documentation for the supported strategy options and their defaults."""
    doc = ""
    for opt, val in strategy_options.items():
        doc += f"     * {opt}: {val[0]}, default {str(val[1])}. \n"
    doc += "\n"
    return doc


def get_options(strategy_options, options, unsupported=None):
    """Get the strategy-specific options or their defaults from user-supplied strategy_options."""
    accepted = list(options.keys()) + ["max_fevals", "time_limit", "x0", "searchspace_construction_options"]
    if unsupported:
        for key in unsupported:
            accepted.remove(key)
    for key in strategy_options:
        if key not in accepted:
            raise ValueError(f"Unrecognized option {key} in strategy_options (allowed: {accepted})")
    assert isinstance(options, dict)
    return [strategy_options.get(opt, default) for opt, (_, default) in options.items()]


class CostFunc:
    """Class encapsulating the CostFunc method."""

    def __init__(
        self,
        searchspace: Searchspace,
        tuning_options,
        runner,
        *,
        scaling=False,
        snap=True,
        return_invalid=False,
        return_raw=None,
        invalid_value=sys.float_info.max,
    ):
        """An abstract method to handle evaluation of configurations.

        Args:
            searchspace: the Searchspace to evaluate on.
            tuning_options: various tuning options.
            runner: the runner to use.
            scaling: whether to internally scale parameter values. Defaults to False.
            snap: whether to snap given configurations to their closests equivalent in the space. Defaults to True.
            return_invalid: whether to return the util.ErrorConfig of an invalid configuration. Defaults to False.
        """
        self.searchspace = searchspace
        self.tuning_options = tuning_options
        if isinstance(self.tuning_options, dict):
            self.tuning_options["max_fevals"] = min(
                tuning_options["max_fevals"] if "max_fevals" in tuning_options else np.inf, searchspace.size
            )
        self.objective = tuning_options.objective
        self.objective_higher_is_better = tuning_options.objective_higher_is_better
        self.constraint_aware = bool(tuning_options.strategy_options.get("constraint_aware"))
        self.runner = runner
        self.scaling = scaling
        self.snap = snap
        self.return_invalid = return_invalid
        self.results = []
        self.budget_spent_fraction = 0.0
        self.invalid_return_value = invalid_value

    def _normalize_and_validate_config(self, x, check_restrictions=True):
        # snap values in x to nearest actual value for each parameter, unscale x if needed
        if self.snap:
            if self.scaling:
                config = unscale_and_snap_to_nearest(x, self.searchspace.tune_params, self.tuning_options.eps)
            else:
                config = snap_to_nearest_config(x, self.searchspace.tune_params)
        else:
            config = x

        is_legal = True

        # else check if this is a legal (non-restricted) configuration
        if check_restrictions:
            is_legal = self.searchspace.is_param_config_valid(tuple(config))

        # Attempt to repare the config
        if not is_legal and self.constraint_aware:
            # attempt to repair
            new_config = unscale_and_snap_to_nearest_valid(x, config, self.searchspace, self.tuning_options.eps)

            if new_config:
                config = new_config
                is_legal = True

        return config, is_legal


    def _run_configs(self, xs, check_restrictions=True):
        """ Takes a list of Euclidian coordinates and evaluates the configurations at those points. """
        self.runner.last_strategy_time = 1000 * (perf_counter() - self.runner.last_strategy_start_time)
        self.runner.start_time = perf_counter() # start framework time

        # error value to return for numeric optimizers that need a numerical value
        logging.debug("_cost_func called")

        # check if max_fevals is reached or time limit is exceeded
        self.budget_spent_fraction = util.check_stop_criterion(self.tuning_options)

        batch_configs = []  # The configs to run
        batch_indices = []  # Where to store result in `final_results``
        final_results = []  # List returned to the user
        benchmark_config = []

        for x in xs:
            config, is_legal = self._normalize_and_validate_config(x, check_restrictions=check_restrictions)
            logging.debug("normalize config: %s -> %s (legal: %s)", str(x), str(config), is_legal)

            if is_legal:
                batch_configs.append(config)
                batch_indices.append(len(final_results))
                final_results.append(None)
                x_int = ",".join([str(i) for i in config])
                benchmark_config.append(x_int not in self.tuning_options.unique_results)
            else:
                result = dict(zip(self.searchspace.tune_params.keys(), config))
                result[self.objective] = util.InvalidConfig()
                final_results.append(result)

        # do not overshoot max_fevals if we can avoid it
        if "max_fevals" in self.tuning_options:
            budget = self.tuning_options.max_fevals - len(self.tuning_options.unique_results)
            if sum(benchmark_config) > budget:
                # find index 'budget'th True value
                last_index = _get_nth_true(benchmark_config, budget)+1
                # mask configs we cannot benchmark
                batch_configs = batch_configs[:last_index]
                batch_indices = batch_indices[:last_index]
                final_results = final_results[:batch_indices[-1]+1]

        # compile and benchmark the batch
        batch_results = self.runner.run(batch_configs, self.tuning_options)
        self.results.extend(batch_results)

        # set in the results array
        for index, result in zip(batch_indices, batch_results):
            final_results[index] = result

        # append to `unique_results`
        for config, result, benchmarked in zip(batch_configs, batch_results, benchmark_config):
            if benchmarked:
                x_int = ",".join([str(i) for i in config])
                if x_int not in self.tuning_options.unique_results:
                    self.tuning_options.unique_results[x_int] = result

        # check again for stop condition
        # this check is necessary because some strategies cannot handle partially completed requests
        # for example when only half of the configs in a population have been evaluated
        self.budget_spent_fraction = util.check_stop_criterion(self.tuning_options)

        # upon returning from this function control will be given back to the strategy, so reset the start time
        self.runner.last_strategy_start_time = perf_counter()
        return final_results

    def eval_all(self, xs, check_restrictions=True):
        """Cost function used by almost all strategies."""
        results = self._run_configs(xs, check_restrictions=check_restrictions)
        return_values = []

        for result in results:
            # get numerical return value, taking optimization direction into account
            return_value = result[self.objective]

            if not isinstance(return_value, util.ErrorConfig):
                # this is a valid configuration, so invert value in case of maximization
                if self.objective_higher_is_better:
                    return_value = -return_value
            else:
                # this is not a valid configuration, replace with float max if needed
                if not self.return_invalid:
                    return_value = sys.float_info.max

            # include raw data in return if requested
            return_values.append(return_value)

        return return_values

    def eval(self, x, check_restrictions=True):
        return self.eval_all([x], check_restrictions=check_restrictions)[0]

    def __call__(self, x, check_restrictions=True):
        return self.eval(x, check_restrictions=check_restrictions)

    def get_start_pos(self):
        """Get starting position for optimization."""
        _, x0, _ = self.get_bounds_x0_eps()
        return x0

    def get_bounds_x0_eps(self):
        """Compute bounds, x0 (the initial guess), and eps."""
        values = list(self.searchspace.tune_params.values())

        if "x0" in self.tuning_options.strategy_options:
            x0 = self.tuning_options.strategy_options.x0
            assert isinstance(x0, (tuple, list)) and len(x0) == len(values), f"Invalid x0: {x0}, expected number of parameters of `tune_params` to match ({len(values)})"
        else:
            x0 = None

        if self.scaling:
            eps = np.amin([1.0 / len(v) for v in values])

            # reducing interval from [0, 1] to [0, eps*len(v)]
            bounds = [(0, eps * len(v)) for v in values]
            if x0:
                # x0 has been supplied by the user, map x0 into [0, eps*len(v)]
                x0 = scale_from_params(x0, self.searchspace.tune_params, eps)
            else:
                # get a valid x0
                pos = list(self.searchspace.get_random_sample(1)[0])
                x0 = scale_from_params(pos, self.searchspace.tune_params, eps)
        else:
            bounds = self.get_bounds()
            if not x0:
                x0 = list(self.searchspace.get_random_sample(1)[0])
            eps = 1

        self.tuning_options["eps"] = eps
        logging.debug("get_bounds_x0_eps called")
        logging.debug("bounds %s", str(bounds))
        logging.debug("x0 %s", str(x0))
        logging.debug("eps %s", str(eps))

        return bounds, x0, eps

    def get_bounds(self):
        """Create a bounds array from the tunable parameters."""
        bounds = []
        for values in self.searchspace.params_values:
            try:
                bounds.append((min(values), max(values)))
            except TypeError:
                # if values are not numbers, use the first and last value as bounds
                bounds.append((values[0], values[-1]))
        return bounds


def _get_nth_true(lst, n):
    # Returns the index of the nth True value in a list
    return [i for i, x in enumerate(lst) if x][n-1]


def setup_method_arguments(method, bounds):
    """Prepare method specific arguments."""
    kwargs = {}
    # pass bounds to methods that support it
    if method in ["L-BFGS-B", "TNC", "SLSQP"]:
        kwargs["bounds"] = bounds
    return kwargs


def setup_method_options(method, tuning_options):
    """Prepare method specific options."""
    kwargs = {}

    # Note that not all methods iterpret maxiter in the same manner
    if "maxiter" in tuning_options.strategy_options:
        maxiter = tuning_options.strategy_options.maxiter
    else:
        maxiter = 100
    kwargs["maxiter"] = maxiter
    if method in ["Nelder-Mead", "Powell"]:
        kwargs["maxfev"] = maxiter
    elif method == "L-BFGS-B":
        kwargs["maxfun"] = maxiter

    # pass eps to methods that support it
    if method in ["CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP"]:
        kwargs["eps"] = tuning_options.eps
    elif method == "COBYLA":
        kwargs["rhobeg"] = tuning_options.eps

    # not all methods support 'disp' option
    if method not in ["TNC"]:
        kwargs["disp"] = tuning_options.verbose

    return kwargs


def snap_to_nearest_config(x, tune_params):
    """Helper func that for each param selects the closest actual value."""
    params = []
    for i, k in enumerate(tune_params.keys()):
        values = tune_params[k]

        # if `x[i]` is in `values`, use that value, otherwise find the closest match
        if x[i] in values:
            idx = values.index(x[i])
        else:
            idx = np.argmin([abs(v - x[i]) for v in values])

        params.append(values[idx])
    return params


def unscale_and_snap_to_nearest(x, tune_params, eps):
    """Helper func that snaps a scaled variable to the nearest config."""
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
    """Helper func to do the inverse of the 'unscale' function."""
    x = np.zeros(len(params))
    for i, v in enumerate(tune_params.values()):
        x[i] = 0.5 * eps + v.index(params[i]) * eps
    return x



def unscale_and_snap_to_nearest_valid(x, params, searchspace, eps):
    """Helper func to snap to the nearest valid configuration"""
    # params is nearest unscaled point, but is not valid
    neighbors = get_neighbors(params, searchspace)

    if neighbors:
        # sort on distance to x
        neighbors.sort(key=lambda y: distance.euclidean(x,scale_from_params(y, searchspace.tune_params, eps)))

        # return closest valid neighbor
        return neighbors[0]

    return []


def get_neighbors(params, searchspace):
    for neighbor_method in ["strictly-adjacent", "adjacent", "Hamming"]:
        neighbors = searchspace.get_neighbors(tuple(params), neighbor_method=neighbor_method)
        if len(neighbors) > 0:
            return neighbors
    return []
