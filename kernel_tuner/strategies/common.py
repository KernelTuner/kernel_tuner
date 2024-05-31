import logging
import sys
from time import perf_counter
import warnings
import ray

import numpy as np

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.util import get_num_devices
from kernel_tuner.runners.ray.remote_actor import RemoteActor
from kernel_tuner.observers.nvml import NVMLObserver, NVMLPowerObserver
from kernel_tuner.observers.pmt import PMTObserver
from kernel_tuner.observers.powersensor import PowerSensorObserver
from kernel_tuner.observers.register import RegisterObserver

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
    return _docstring_template.replace("$NAME$", name).replace("$STRAT_OPT$", make_strategy_options_doc(strategy_options))


def make_strategy_options_doc(strategy_options):
    """Generate documentation for the supported strategy options and their defaults."""
    doc = ""
    for opt, val in strategy_options.items():
        doc += f"     * {opt}: {val[0]}, default {str(val[1])}. \n"
    doc += "\n"
    return doc


def get_options(strategy_options, options):
    """Get the strategy-specific options or their defaults from user-supplied strategy_options."""
    accepted = list(options.keys()) + ["max_fevals", "time_limit", "ensemble", "candidates", "candidate", "population", 
                                       "maxiter", "lsd", "popsize", "alsd", "split_searchspace"]
    for key in strategy_options:
        if key not in accepted:
            raise ValueError(f"Unrecognized option {key} in strategy_options")
    assert isinstance(options, dict)
    return [strategy_options.get(opt, default) for opt, (_, default) in options.items()]


class CostFunc:
    def __init__(self, searchspace: Searchspace, tuning_options, runner, *, scaling=False, snap=True):
        self.runner = runner
        self.tuning_options = tuning_options
        self.snap = snap
        self.scaling = scaling
        self.searchspace = searchspace
        self.results = []

    def __call__(self, x, check_restrictions=True):
        """Cost function used by almost all strategies."""
        self.runner.last_strategy_time = 1000 * (perf_counter() - self.runner.last_strategy_start_time)

        # error value to return for numeric optimizers that need a numerical value
        logging.debug('_cost_func called')
        logging.debug('x: ' + str(x))

        # check if max_fevals is reached or time limit is exceeded
        util.check_stop_criterion(self.tuning_options)

        x_list = [x] if self._is_single_configuration(x) else x
        configs = [self._prepare_config(cfg) for cfg in x_list]
        
        legal_configs = configs
        illegal_results = []
        if check_restrictions and self.searchspace.restrictions:
            legal_configs, illegal_results = self._get_legal_configs(configs)
        
        final_results = self._evaluate_configs(legal_configs) if len(legal_configs) > 0 else []
        # get numerical return values, taking optimization direction into account
        all_results = final_results + illegal_results
        return_values = []
        for result in all_results:
            return_value = result[self.tuning_options.objective] or sys.float_info.max
            return_values.append(return_value if not self.tuning_options.objective_higher_is_better else -return_value)
        
        if len(return_values) == 1:
            return return_values[0]
        return return_values
    
    def _is_single_configuration(self, x):
        """
        Determines if the input is a single configuration based on its type and composition.
        
        Parameters:
            x: The input to check, which can be an int, float, numpy array, list, or tuple.

        Returns:
            bool: True if `x` is a single configuration, which includes being a singular int or float, 
                a numpy array of ints or floats, or a list or tuple where all elements are ints or floats.
                Otherwise, returns False.
        """
        if isinstance(x, (int, float)):
            return True
        if isinstance(x, np.ndarray):
            return x.dtype.kind in 'if'  # Checks for data type being integer ('i') or float ('f')
        if isinstance(x, (list, tuple)):
            return all(isinstance(item, (int, float)) for item in x)
        return False
    
    def _prepare_config(self, x):
        """
        Prepare a single configuration by snapping to nearest values and/or scaling.

        Args:
            x (list): The input configuration to be prepared.

        Returns:
            list: The prepared configuration.

        """
        if self.snap:
            if self.scaling:
                params = unscale_and_snap_to_nearest(x, self.searchspace.tune_params, self.tuning_options.eps)
            else:
                params = snap_to_nearest_config(x, self.searchspace.tune_params)
        else:
            params = x
        return params
    
    def _get_legal_configs(self, configs):
        """
        Filters and categorizes configurations into legal and illegal based on defined restrictions. 
        Configurations are checked against restrictions; illegal ones are modified to indicate an invalid state and 
        included in the results. Legal configurations are collected and returned for potential use.

        Parameters:
            configs (list of tuple): Configurations to be checked, each represented as a tuple of parameter values.

        Returns:
            tuple: A pair containing a list of legal configurations and a list of results with illegal configurations marked.
        """
        results = []
        legal_configs = []
        for config in configs:
            params_dict = dict(zip(self.searchspace.tune_params.keys(), config))
            legal = util.check_restrictions(self.searchspace.restrictions, params_dict, self.tuning_options.verbose)
            if not legal:
                params_dict[self.tuning_options.objective] = util.InvalidConfig()
                results.append(params_dict)
            else:
                legal_configs.append(config)
        return legal_configs, results
    
    def _evaluate_configs(self, configs):
        """
        Evaluate and manage configurations based on tuning options. Results are sorted by timestamp to maintain 
        order during parallel processing. The function ensures no duplicates in results and checks for stop criteria 
        post-processing. Strategy start time is updated upon completion.

        Parameters:
            configs (list): Configurations to be evaluated.

        Returns:
            list of dict: Processed results of the evaluations.
        """
        results = self.runner.run(configs, self.tuning_options)
        # sort based on timestamp, needed because of parallel tuning of populations and restrospective stop criterion check
        if "timestamp" in results[0]:
            results.sort(key=lambda x: x['timestamp'])

        final_results = []
        for result in results:
            config = tuple(result[key] for key in self.tuning_options.tune_params if key in result)
            x_int = ",".join([str(i) for i in config])
            # append to tuning results
            if x_int not in self.tuning_options.unique_results:
                self.tuning_options.unique_results[x_int] = result
                # check retrospectively if max_fevals is reached or time limit is exceeded within the results
                util.check_stop_criterion(self.tuning_options)
            final_results.append(result)
            # in case of stop creterion reached, save the results so far
            self.results.append(result)

        self.results.extend(final_results)
        # upon returning from this function control will be given back to the strategy, so reset the start time
        self.runner.last_strategy_start_time = perf_counter()

        return final_results

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
        logging.debug('get_bounds_x0_eps called')
        logging.debug('bounds ' + str(bounds))
        logging.debug('x0 ' + str(x0))
        logging.debug('eps ' + str(eps))

        return bounds, x0, eps

    def get_bounds(self):
        """Create a bounds array from the tunable parameters."""
        bounds = []
        for values in self.searchspace.tune_params.values():
            sorted_values = np.sort(values)
            bounds.append((sorted_values[0], sorted_values[-1]))
        return bounds


def setup_method_arguments(method, bounds):
    """Prepare method specific arguments."""
    kwargs = {}
    # pass bounds to methods that support it
    if method in ["L-BFGS-B", "TNC", "SLSQP"]:
        kwargs['bounds'] = bounds
    return kwargs


def setup_method_options(method, tuning_options):
    """Prepare method specific options."""
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
        x[i] = 0.5 * eps + v.index(params[i])*eps
    return x

def check_num_devices(ensemble_size: int, simulation_mode: bool, runner):
    
    num_devices = get_num_devices(runner.kernel_source.lang, simulation_mode=simulation_mode)
    if num_devices < ensemble_size:
         warnings.warn("Number of devices is less than the number of strategies in the ensemble. Some strategies will wait until devices are available.", UserWarning)

def create_actor_on_device(kernel_source, kernel_options, device_options, iterations, observers, cache_manager, simulation_mode, id):
    # Check if Ray is initialized, raise an error if not
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Initialize Ray before creating an actor (remember to include resources).")

    if simulation_mode:
        resource_options = {"num_cpus": 1}
    else:
        resource_options = {"num_gpus": 1}
    
    observers_type_and_arguments = []
    if observers is not None:
        # observers can't be pickled so we will re-initialize them in the actors
        # observers related to backends will be initialized once we call the device interface inside the actor, that is why we skip them here
        for i, observer in enumerate(observers):
            if isinstance(observer, (NVMLObserver, NVMLPowerObserver, PMTObserver, PowerSensorObserver)):
                observers_type_and_arguments.append((observer.__class__, observer.init_arguments))
            if isinstance(observer, RegisterObserver):
                observers_type_and_arguments.append((observer.__class__, []))
    
    # Create the actor with the specified options and resources
    return RemoteActor.options(**resource_options).remote(kernel_source, 
                                                            kernel_options, 
                                                            device_options, 
                                                            iterations, 
                                                            observers_type_and_arguments=observers_type_and_arguments,
                                                            cache_manager=cache_manager,
                                                            simulation_mode=simulation_mode,
                                                            id=id)

def initialize_ray():
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(include_dashboard=True, ignore_reinit_error=True)

