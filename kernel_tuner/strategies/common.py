"""Module for functionality that is commonly used throughout the strategies."""

import logging
import numbers
import sys
from time import perf_counter

import numpy as np

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


def get_options(strategy_options, options):
    """Get the strategy-specific options or their defaults from user-supplied strategy_options."""
    accepted = list(options.keys()) + ["max_fevals", "time_limit"]
    for key in strategy_options:
        if key not in accepted:
            raise ValueError(f"Unrecognized option {key} in strategy_options")
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
        encode_non_numeric=False,
        return_invalid=False,
        return_raw=None,
    ):
        """An abstract method to handle evaluation of configurations.

        Args:
            searchspace: the Searchspace to evaluate on.
            tuning_options: various tuning options.
            runner: the runner to use.
            scaling: whether to internally scale parameter values. Defaults to False.
            snap: whether to snap given configurations to their closests equivalent in the space. Defaults to True.
            encode_non_numeric: whether to externally encode non-numeric parameter values. Defaults to False.
            return_invalid: whether to return the util.ErrorConfig of an invalid configuration. Defaults to False.
            return_raw: returns (result, results[raw]). Key inferred from objective if set to True. Defaults to None.
        """
        self.runner = runner
        self.snap = snap
        self.scaling = scaling
        self.encode_non_numeric = encode_non_numeric
        self.return_invalid = return_invalid
        self.return_raw = return_raw
        if return_raw is True:
            self.return_raw = f"{tuning_options['objective']}s"
        self.searchspace = searchspace
        self.tuning_options = tuning_options
        if isinstance(self.tuning_options, dict):
            self.tuning_options["max_fevals"] = min(
                tuning_options["max_fevals"] if "max_fevals" in tuning_options else np.inf, searchspace.size
            )
        self.results = []

        # if enabled, encode non-numeric parameter values as a numeric value
        if self.encode_non_numeric:
            self._map_param_to_encoded = {}
            self._map_encoded_to_param = {}
            self.encoded_params_values = []
            for i, param_values in enumerate(self.searchspace.params_values):
                encoded_values = param_values
                if not all(isinstance(v, numbers.Real) for v in param_values):
                    encoded_values = np.arange(
                        len(param_values)
                    )  # NOTE when changing this, adjust the rounding in encoded_to_params
                    self._map_param_to_encoded[i] = dict(zip(param_values, encoded_values))
                    self._map_encoded_to_param[i] = dict(zip(encoded_values, param_values))
                self.encoded_params_values.append(encoded_values)

    def __call__(self, x, check_restrictions=True):
        """Cost function used by almost all strategies."""
        self.runner.last_strategy_time = 1000 * (perf_counter() - self.runner.last_strategy_start_time)
        if self.encode_non_numeric:
            x = self.encoded_to_params(x)

        # error value to return for numeric optimizers that need a numerical value
        logging.debug("_cost_func called")
        logging.debug("x: %s", str(x))

        # check if max_fevals is reached or time limit is exceeded
        util.check_stop_criterion(self.tuning_options)

        # snap values in x to nearest actual value for each parameter, unscale x if needed
        if self.snap:
            if self.scaling:
                params = unscale_and_snap_to_nearest(x, self.searchspace.tune_params, self.tuning_options.eps)
            else:
                params = snap_to_nearest_config(x, self.searchspace.tune_params)
        else:
            params = x
        logging.debug("params %s", str(params))

        legal = True
        result = {}
        x_int = ",".join([str(i) for i in params])

        # else check if this is a legal (non-restricted) configuration
        if check_restrictions and self.searchspace.restrictions:
            params_dict = dict(zip(self.searchspace.tune_params.keys(), params))
            legal = util.check_restrictions(self.searchspace.restrictions, params_dict, self.tuning_options.verbose)
            if not legal:
                result = params_dict
                result[self.tuning_options.objective] = util.InvalidConfig()

        if legal:
            # compile and benchmark this instance
            res = self.runner.run([params], self.tuning_options)
            result = res[0]

            # append to tuning results
            if x_int not in self.tuning_options.unique_results:
                self.tuning_options.unique_results[x_int] = result

            self.results.append(result)

            # upon returning from this function control will be given back to the strategy, so reset the start time
            self.runner.last_strategy_start_time = perf_counter()

        # get numerical return value, taking optimization direction into account
        if self.return_invalid:
            return_value = result[self.tuning_options.objective]
        else:
            return_value = result[self.tuning_options.objective] or sys.float_info.max
        return_value = -return_value if self.tuning_options.objective_higher_is_better else return_value

        # include raw data in return if requested
        if self.return_raw is not None:
            try:
                return return_value, result[self.return_raw]
            except KeyError:
                return return_value, [np.nan]

        return return_value

    def get_bounds_x0_eps(self):
        """Compute bounds, x0 (the initial guess), and eps."""
        values = list(self.searchspace.tune_params.values())

        if "x0" in self.tuning_options.strategy_options:
            x0 = self.tuning_options.strategy_options.x0
        else:
            x0 = None

        if self.scaling:
            eps = np.amin([1.0 / len(v) for v in values])

            # reducing interval from [0, 1] to [0, eps*len(v)]
            bounds = [(0, eps * len(v)) for v in values]
            if x0:
                # x0 has been supplied by the user, map x0 into [0, eps*len(v)]
                x0 = scale_from_params(x0, self.tuning_options, eps)
            else:
                # get a valid x0
                pos = list(self.searchspace.get_random_sample(1)[0])
                x0 = scale_from_params(pos, self.searchspace.tune_params, eps)
        else:
            bounds = self.get_bounds()
            if not x0:
                x0 = [(min_v + max_v) / 2.0 for (min_v, max_v) in bounds]
            eps = 1e9
            for v_list in values:
                if len(v_list) > 1:
                    vals = np.sort(v_list)
                    eps = min(eps, np.amin(np.gradient(vals)))

        self.tuning_options["eps"] = eps
        logging.debug("get_bounds_x0_eps called")
        logging.debug("bounds %s", str(bounds))
        logging.debug("x0 %s", str(x0))
        logging.debug("eps %s", str(eps))

        return bounds, x0, eps

    def get_bounds(self):
        """Create a bounds array from the tunable parameters."""
        bounds = []
        for values in self.encoded_params_values if self.encode_non_numeric else self.searchspace.params_values:
            bounds.append((min(values), max(values)))
        return bounds

    def encoded_to_params(self, config):
        """Convert from an encoded configuration to the real parameters."""
        if not self.encode_non_numeric:
            raise ValueError("'encode_non_numeric' must be set to true to use this function.")
        params = []
        for i, v in enumerate(config):
            # params.append(self._map_encoded_to_param[i][v] if i in self._map_encoded_to_param else v)
            if i in self._map_encoded_to_param:
                encoding = self._map_encoded_to_param[i]
                if v in encoding:
                    param = encoding[v]
                elif isinstance(v, float):
                    # try to resolve a rounding error due to floating point arithmetic / continous solver
                    param = encoding[round(v)]
                else:
                    raise ValueError(f"Encoded value {v} not found in {self._map_encoded_to_param[i]}")
            else:
                param = v
            params.append(param)
        assert len(params) == len(config)
        return params

    def params_to_encoded(self, config):
        """Convert from a parameter configuration to the encoded configuration."""
        if not self.encode_non_numeric:
            raise ValueError("'encode_non_numeric' must be set to true to use this function.")
        encoded = []
        for i, v in enumerate(config):
            encoded.append(self._map_param_to_encoded[i][v] if i in self._map_param_to_encoded else v)
        assert len(encoded) == len(config)
        return encoded


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
