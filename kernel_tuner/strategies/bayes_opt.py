""" Bayesian Optimization implementation from the thesis by Willemsen """
from copy import deepcopy
from random import randint, shuffle
import itertools
import warnings
import time
from typing import Tuple

import numpy as np
from scipy.stats import norm

# BO imports
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
    from sklearn.exceptions import ConvergenceWarning
    from skopt.sampler import Lhs
    bayes_opt_present = True
except ImportError:
    bayes_opt_present = False

from kernel_tuner.strategies import minimize
from kernel_tuner import util

supported_methods = ["poi", "ei", "lcb", "lcb-srinivas", "multi", "multi-advanced", "multi-fast"]


def generate_normalized_param_dicts(tune_params: dict, eps: float) -> Tuple[dict, dict]:
    """ Generates normalization and denormalization dictionaries """
    original_to_normalized = dict()
    normalized_to_original = dict()
    for param_name in tune_params.keys():
        original_to_normalized_dict = dict()
        normalized_to_original_dict = dict()
        for value_index, value in enumerate(tune_params[param_name]):
            normalized_value = eps * value_index + 0.5 * eps
            normalized_to_original_dict[normalized_value] = value
            original_to_normalized_dict[value] = normalized_value
        original_to_normalized[param_name] = original_to_normalized_dict
        normalized_to_original[param_name] = normalized_to_original_dict
    return original_to_normalized, normalized_to_original


def normalize_parameter_space(param_space: list, tune_params: dict, normalized: dict) -> list:
    """ Normalize the parameter space given a normalization dictionary """
    keys = list(tune_params.keys())
    param_space_normalized = list(tuple(normalized[keys[i]][v] for i, v in enumerate(params)) for params in param_space)
    return param_space_normalized


def prune_parameter_space(parameter_space, tuning_options, tune_params, normalize_dict):
    """ Pruning of the parameter space to remove dimensions that have a constant parameter """
    pruned_tune_params_mask = list()
    removed_tune_params = list()
    param_names = list(tune_params.keys())
    for index, key in enumerate(tune_params.keys()):
        pruned_tune_params_mask.append(len(tune_params[key]) > 1)
        if len(tune_params[key]) > 1:
            removed_tune_params.append(None)
        else:
            value = tune_params[key][0]
            normalized = normalize_dict[param_names[index]][value]
            removed_tune_params.append(normalized)
    if 'verbose' in tuning_options and tuning_options.verbose is True and len(tune_params.keys()) != sum(pruned_tune_params_mask):
        print(f"Number of parameters (dimensions): {len(tune_params.keys())}, after pruning: {sum(pruned_tune_params_mask)}")
    parameter_space = list(tuple(itertools.compress(param_config, pruned_tune_params_mask)) for param_config in parameter_space)
    return parameter_space, removed_tune_params


def tune(runner, kernel_options, device_options, tuning_options):
    """ Find the best performing kernel configuration in the parameter space

    :params runner: A runner from kernel_tuner.runners
    :type runner: kernel_tuner.runner

    :param kernel_options: A dictionary with all options for the kernel.
    :type kernel_options: kernel_tuner.interface.Options

    :param device_options: A dictionary with all options for the device
        on which the kernel should be tuned.
    :type device_options: kernel_tuner.interface.Options

    :param tuning_options: A dictionary with all options regarding the tuning
        process. Allows setting hyperparameters via the strategy_options key.
    :type tuning_options: kernel_tuner.interface.Options

    :returns: A list of dictionaries for executed kernel configurations and their
        execution times. And a dictionary that contains a information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    max_fevals = tuning_options.strategy_options.get("max_fevals", 100)
    prune_parameterspace = tuning_options.strategy_options.get("pruneparameterspace", True)
    if not bayes_opt_present:
        raise ImportError("Error: optional dependencies for Bayesian Optimization not installed, please install scikit-learn and scikit-optimize")

    # epsilon for scaling should be the evenly spaced distance between the largest set of parameter options in an interval [0,1]
    tune_params = tuning_options.tune_params
    tuning_options["scaling"] = True
    _, _, eps = minimize.get_bounds_x0_eps(tuning_options)

    # compute cartesian product of all tunable parameters
    parameter_space = itertools.product(*tune_params.values())

    # check for search space restrictions
    if tuning_options.restrictions is not None:
        tuning_options.verbose = False
    parameter_space = filter(lambda p: util.config_valid(p, tuning_options, runner.dev.max_threads), parameter_space)
    parameter_space = list(parameter_space)
    if len(parameter_space) < 1:
        raise ValueError("Empty parameterspace after restrictionscheck. Restrictionscheck is possibly too strict.")
    if len(parameter_space) == 1:
        raise ValueError(f"Only one configuration after restrictionscheck. Restrictionscheck is possibly too strict. Configuration: {parameter_space[0]}")

    # normalize search space to [0,1]
    normalize_dict, denormalize_dict = generate_normalized_param_dicts(tune_params, eps)
    parameter_space = normalize_parameter_space(parameter_space, tune_params, normalize_dict)

    # prune the parameter space to remove dimensions that have a constant parameter
    if prune_parameterspace:
        parameter_space, removed_tune_params = prune_parameter_space(parameter_space, tuning_options, tune_params, normalize_dict)
    else:
        parameter_space = list(parameter_space)
        removed_tune_params = [None] * len(tune_params.keys())

    # initialize and optimize
    bo = BayesianOptimization(parameter_space, removed_tune_params, kernel_options, tuning_options, normalize_dict, denormalize_dict, runner)
    results = bo.optimize(max_fevals)

    return results, runner.dev.get_environment()


class BayesianOptimization():

    def __init__(self, searchspace: list, removed_tune_params: list, kernel_options: dict, tuning_options: dict, normalize_dict: dict, denormalize_dict: dict,
                 runner, opt_direction='min'):
        time_start = time.perf_counter_ns()

        # supported hyperparameter values
        self.supported_cov_kernels = ["constantrbf", "rbf", "matern32", "matern52"]
        self.supported_methods = supported_methods
        self.supported_sampling_methods = ["random", "lhs"]
        self.supported_sampling_criterion = ["correlation", "ratio", "maximin", None]

        def get_hyperparam(name: str, default, supported_values=list()):
            value = tuning_options.strategy_options.get(name, default)
            if len(supported_values) > 0 and value not in supported_values:
                raise ValueError(f"'{name}' is set to {value}, but must be one of {supported_values}")
            return value

        # get hyperparameters
        cov_kernel_name = get_hyperparam("covariancekernel", "matern32", self.supported_cov_kernels)
        cov_kernel_lengthscale = get_hyperparam("covariancelengthscale", 1.5)
        acquisition_function = get_hyperparam("method", "multi-advanced", self.supported_methods)
        acq = acquisition_function
        acq_params = get_hyperparam("methodparams", {})
        multi_af_names = get_hyperparam("multi_af_names", ['ei', 'poi', 'lcb'])
        self.multi_afs_discount_factor = get_hyperparam("multi_af_discount_factor", 0.65 if acq == 'multi' else 0.95)
        self.multi_afs_required_improvement_factor = get_hyperparam("multi_afs_required_improvement_factor", 0.15 if acq == 'multi-advanced-precise' else 0.1)
        self.num_initial_samples = get_hyperparam("popsize", 20)
        self.sampling_method = get_hyperparam("samplingmethod", "lhs", self.supported_sampling_methods)
        self.sampling_crit = get_hyperparam("samplingcriterion", 'maximin', self.supported_sampling_criterion)
        self.sampling_iter = get_hyperparam("samplingiterations", 1000)

        # set acquisition function hyperparameter defaults where missing
        if 'explorationfactor' not in acq_params:
            acq_params['explorationfactor'] = 'CV'
        if 'zeta' not in acq_params:
            acq_params['zeta'] = 1
        if 'skip_duplicate_after' not in acq_params:
            acq_params['skip_duplicate_after'] = 5

        # set arguments
        self.kernel_options = kernel_options
        self.tuning_options = tuning_options
        self.tune_params = tuning_options.tune_params
        self.param_names = list(self.tune_params.keys())
        self.normalized_dict = normalize_dict
        self.denormalized_dict = denormalize_dict
        self.runner = runner
        self.max_threads = runner.dev.max_threads
        self.log_timings = False

        # set optimization constants
        self.invalid_value = 1e20
        self.opt_direction = opt_direction
        if opt_direction == 'min':
            self.worst_value = np.PINF
            self.argopt = np.argmin
        elif opt_direction == 'max':
            self.worst_value = np.NINF
            self.argopt = np.argmax
        else:
            raise ValueError("Invalid optimization direction '{}'".format(opt_direction))

        # set the acquisition function and surrogate model
        self.optimize = self.__optimize
        self.af_name = acquisition_function
        self.af_params = acq_params
        self.multi_afs = list(self.get_af_by_name(af_name) for af_name in multi_af_names)
        self.set_acquisition_function(acquisition_function)
        self.set_surrogate_model(cov_kernel_name, cov_kernel_lengthscale)

        # set remaining values
        self.results = []
        self.__searchspace = searchspace
        self.removed_tune_params = removed_tune_params
        self.searchspace_size = len(self.searchspace)
        self.num_dimensions = len(self.dimensions())
        self.__current_optimum = self.worst_value
        self.cv_norm_maximum = None
        self.fevals = 0
        self.__visited_num = 0
        self.__visited_valid_num = 0
        self.__visited_searchspace_indices = [False] * self.searchspace_size
        self.__observations = [np.NaN] * self.searchspace_size
        self.__valid_observation_indices = [False] * self.searchspace_size
        self.__valid_params = list()
        self.__valid_observations = list()
        self.unvisited_cache = self.unvisited()
        time_setup = time.perf_counter_ns()
        self.error_message_searchspace_fully_observed = "The search space has been fully observed"

        # take initial sample
        if self.num_initial_samples > 0:
            self.initial_sample()
            time_initial_sample = time.perf_counter_ns()

        # print the timings
        if self.log_timings:
            time_taken_setup = round(time_setup - time_start, 3) / 1000
            time_taken_initial_sample = round(time_initial_sample - time_setup, 3) / 1000
            time_taken_total = round(time_initial_sample - time_start, 3) / 1000
            print(f"Initialization | total time: {time_taken_total} | Setup: {time_taken_setup} | Initial sample: {time_taken_initial_sample}", flush=True)

    @property
    def searchspace(self):
        return self.__searchspace

    @property
    def observations(self):
        return self.__observations

    @property
    def current_optimum(self):
        return self.__current_optimum

    @current_optimum.setter
    def current_optimum(self, value: float):
        self.__current_optimum = value

    def is_better_than(self, a: float, b: float) -> bool:
        """ Determines which one is better depending on optimization direction """
        return a < b if self.opt_direction == 'min' else a > b

    def is_not_visited(self, index: int) -> bool:
        """ Returns whether a searchspace index has not been visited """
        return not self.__visited_searchspace_indices[index]

    def is_valid(self, observation: float) -> bool:
        """ Returns whether an observation is valid """
        return not (observation == None or observation == self.invalid_value or observation == np.NaN)

    def get_af_by_name(self, name: str):
        """ Get the basic acquisition functions by their name """
        basic_af_names = ['ei', 'poi', 'lcb']
        if name == 'ei':
            return self.af_expected_improvement
        elif name == 'poi':
            return self.af_probability_of_improvement
        elif name == 'lcb':
            return self.af_lower_confidence_bound
        raise ValueError(f"{name} not in {basic_af_names}")

    def set_acquisition_function(self, acquisition_function: str):
        """ Set the acquisition function """
        if acquisition_function == 'poi':
            self.__af = self.af_probability_of_improvement
        elif acquisition_function == 'ei':
            self.__af = self.af_expected_improvement
        elif acquisition_function == 'lcb':
            self.__af = self.af_lower_confidence_bound
        elif acquisition_function == 'lcb-srinivas':
            self.__af = self.af_lower_confidence_bound_srinivas
        elif acquisition_function == 'random':
            self.__af = self.af_random
        elif acquisition_function == 'multi':
            self.optimize = self.__optimize_multi
        elif acquisition_function == 'multi-advanced':
            self.optimize = self.__optimize_multi_advanced
        elif acquisition_function == 'multi-fast':
            self.optimize = self.__optimize_multi_fast
        else:
            raise ValueError("Acquisition function must be one of {}, is {}".format(self.supported_methods, acquisition_function))

    def set_surrogate_model(self, cov_kernel_name: str, cov_kernel_lengthscale: float):
        """ Set the surrogate model with a covariance function and lengthscale """
        if cov_kernel_name == "constantrbf":
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(cov_kernel_lengthscale, length_scale_bounds="fixed")
        elif cov_kernel_name == "rbf":
            kernel = RBF(length_scale=cov_kernel_lengthscale, length_scale_bounds="fixed")
        elif cov_kernel_name == "matern32":
            kernel = Matern(length_scale=cov_kernel_lengthscale, nu=1.5, length_scale_bounds="fixed")
        elif cov_kernel_name == "matern52":
            kernel = Matern(length_scale=cov_kernel_lengthscale, nu=2.5, length_scale_bounds="fixed")
        else:
            raise ValueError("Acquisition function must be one of {}, is {}".format(self.supported_cov_kernels, cov_kernel_name))
        self.__model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)    # maybe change alpha to a higher value such as 1e-5?

    def valid_params_observations(self) -> Tuple[list, list]:
        """ Returns a list of valid observations and their parameter configurations """
        # if you do this every iteration, better keep it as cache and update in update_after_evaluation
        params = list()
        observations = list()
        for index, valid in enumerate(self.__valid_observation_indices):
            if valid is True:
                params.append(self.searchspace[index])
                observations.append(self.observations[index])
        return params, observations

    def unvisited(self) -> list:
        """ Returns a list of unvisited parameter configurations - attention: cached version exists! """
        params = list(self.searchspace[index] for index, visited in enumerate(self.__visited_searchspace_indices) if visited is False)
        return params

    def find_param_config_index(self, param_config: tuple) -> int:
        """ Find a parameter config index in the search space if it exists """
        return self.searchspace.index(param_config)

    def find_param_config_unvisited_index(self, param_config: tuple) -> int:
        """ Find a parameter config index in the unvisited cache if it exists """
        return self.unvisited_cache.index(param_config)

    def normalize_param_config(self, param_config: tuple) -> tuple:
        """ Normalizes a parameter configuration """
        normalized = tuple(self.normalized_dict[self.param_names[index]][param_value] for index, param_value in enumerate(param_config))
        return normalized

    def denormalize_param_config(self, param_config: tuple) -> tuple:
        """ Denormalizes a parameter configuration """
        denormalized = tuple(self.denormalized_dict[self.param_names[index]][param_value] for index, param_value in enumerate(param_config))
        return denormalized

    def unprune_param_config(self, param_config: tuple) -> tuple:
        """ In case of pruned dimensions, adds the removed dimensions back in the param config """
        unpruned = list()
        pruned_count = 0
        for removed in self.removed_tune_params:
            if removed is not None:
                unpruned.append(removed)
            else:
                unpruned.append(param_config[pruned_count])
                pruned_count += 1
        return tuple(unpruned)

    def update_after_evaluation(self, observation: float, index: int, param_config: tuple):
        """ Adjust the visited and valid index records accordingly """
        validity = self.is_valid(observation)
        self.__visited_num += 1
        self.__observations[index] = observation
        self.__visited_searchspace_indices[index] = True
        del self.unvisited_cache[self.find_param_config_unvisited_index(param_config)]
        self.__valid_observation_indices[index] = validity
        if validity is True:
            self.__visited_valid_num += 1
            self.__valid_params.append(param_config)
            self.__valid_observations.append(observation)
            if self.is_better_than(observation, self.current_optimum):
                self.current_optimum = observation

    def predict(self, x) -> Tuple[float, float]:
        """ Returns a mean and standard deviation predicted by the surrogate model for the parameter configuration """
        return self.__model.predict([x], return_std=True)

    def predict_list(self, lst: list) -> Tuple[list, list, list]:
        """ Returns a list of means and standard deviations predicted by the surrogate model for the parameter configurations, and separate lists of means and standard deviations """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, std = self.__model.predict(lst, return_std=True)
            return list(zip(mu, std)), mu, std

    def fit_observations_to_model(self):
        """ Update the model based on the current list of observations """
        self.__model.fit(self.__valid_params, self.__valid_observations)

    def evaluate_objective_function(self, param_config: tuple) -> float:
        """ Evaluates the objective function """
        param_config = self.unprune_param_config(param_config)
        denormalized_param_config = self.denormalize_param_config(param_config)
        if not util.config_valid(denormalized_param_config, self.tuning_options, self.max_threads):
            return self.invalid_value
        val = minimize._cost_func(param_config, self.kernel_options, self.tuning_options, self.runner, self.results)
        self.fevals += 1
        return val

    def dimensions(self) -> list:
        """ List of parameter values per parameter """
        return self.tune_params.values()

    def draw_random_sample(self) -> Tuple[list, int]:
        """ Draw a random sample from the unvisited parameter configurations """
        if len(self.unvisited_cache) < 1:
            raise ValueError("Searchspace exhausted during random sample draw as no valid configurations were found")
        index = randint(0, len(self.unvisited_cache) - 1)    # NOSONAR
        param_config = self.unvisited_cache[index]
        actual_index = self.find_param_config_index(param_config)
        return param_config, actual_index

    def draw_latin_hypercube_samples(self, num_samples: int) -> list:
        """ Draws an LHS-distributed sample from the search space """
        if self.searchspace_size < num_samples:
            raise ValueError("Can't sample more than the size of the search space")
        if self.sampling_crit is None:
            lhs = Lhs(lhs_type="centered", criterion=None)
        else:
            lhs = Lhs(lhs_type="classic", criterion=self.sampling_crit, iterations=self.sampling_iter)
        param_configs = lhs.generate(self.dimensions(), num_samples)
        indices = list()
        normalized_param_configs = list()
        for i in range(len(param_configs) - 1):
            try:
                param_config = self.normalize_param_config(param_configs[i])
                index = self.find_param_config_index(param_config)
                indices.append(index)
                normalized_param_configs.append(param_config)
            except ValueError:
                """ Due to search space restrictions, the search space may not be an exact cartesian product of the tunable parameter values.
                It is thus possible for LHS to generate a parameter combination that is not in the actual searchspace, which must be skipped. """
                continue
        return list(zip(normalized_param_configs, indices))

    def initial_sample(self):
        """ Draws an initial sample using random sampling """
        if self.num_initial_samples <= 0:
            raise ValueError("At least one initial sample is required")
        if self.sampling_method == 'lhs':
            samples = self.draw_latin_hypercube_samples(self.num_initial_samples)
        elif self.sampling_method == 'random':
            samples = list()
        else:
            raise ValueError("Sampling method must be one of {}, is {}".format(self.supported_sampling_methods, self.sampling_method))
        # collect the samples
        collected_samples = 0
        for params, index in samples:
            observation = self.evaluate_objective_function(params)
            self.update_after_evaluation(observation, index, params)
            if self.is_valid(observation):
                collected_samples += 1
        # collect the remainder of the samples
        while collected_samples < self.num_initial_samples:
            params, index = self.draw_random_sample()
            observation = self.evaluate_objective_function(params)
            self.update_after_evaluation(observation, index, params)
            # check for validity to avoid having no actual initial samples
            if self.is_valid(observation):
                collected_samples += 1
        self.fit_observations_to_model()
        _, _, std = self.predict_list(self.unvisited_cache)
        self.initial_sample_mean = np.mean(self.__valid_observations)
        # Alternatively:
        # self.initial_sample_std = np.std(self.__valid_observations)
        # self.initial_sample_mean = np.mean(predictions)
        self.initial_std = np.mean(std)
        self.cv_norm_maximum = self.initial_std

    def contextual_variance(self, std: list):
        """ Contextual improvement to decide explore / exploit, based on CI proposed by (Jasrasaria, 2018) """
        if not self.af_params['explorationfactor'] == 'CV':
            return None
        if self.opt_direction == 'min':
            if self.current_optimum == self.worst_value:
                return 0.01
            if self.current_optimum <= 0:
                # doesn't work well for minimization beyond 0, should that even be a thing?
                return abs(np.mean(std) / self.current_optimum)
            improvement_over_initial_sample = self.initial_sample_mean / self.current_optimum
            cv = np.mean(std) / improvement_over_initial_sample
            # normalize if available
            if self.cv_norm_maximum:
                cv = cv / self.cv_norm_maximum
            return cv
        return np.mean(std) / self.current_optimum

    def __optimize(self, max_fevals):
        """ Find the next best candidate configuration(s), evaluate those and update the model accordingly """
        while self.fevals < max_fevals:
            if self.__visited_num >= self.searchspace_size:
                raise ValueError(self.error_message_searchspace_fully_observed)
            predictions, _, std = self.predict_list(self.unvisited_cache)
            hyperparam = self.contextual_variance(std)
            list_of_acquisition_values = self.__af(predictions, hyperparam)
            # afterwards select the best AF value
            best_af = self.argopt(list_of_acquisition_values)
            candidate_params = self.unvisited_cache[best_af]
            candidate_index = self.find_param_config_index(candidate_params)
            observation = self.evaluate_objective_function(candidate_params)
            self.update_after_evaluation(observation, candidate_index, candidate_params)
            self.fit_observations_to_model()
        return self.results

    def __optimize_multi(self, max_fevals):
        """ Optimize with a portfolio of multiple acquisition functions. Predictions are always only taken once. Skips AFs if they suggest X/max_evals duplicates in a row, prefers AF with best discounted average. """
        if self.opt_direction != 'min':
            raise ValueError(f"Optimization direction must be minimization ('min'), is {self.opt_direction}")
        # calculate how many times an AF can suggest a duplicate candidate before the AF is skipped
        # skip_duplicates_fraction = self.af_params['skip_duplicates_fraction']
        # skip_if_duplicate_n_times = int(min(max(round(skip_duplicates_fraction * max_fevals), 3), max_fevals))
        skip_if_duplicate_n_times = self.af_params['skip_duplicate_after']
        discount_factor = self.multi_afs_discount_factor
        # setup the registration of duplicates and runtimes
        duplicate_count_template = [0 for _ in range(skip_if_duplicate_n_times)]
        duplicate_candidate_af_count = list(deepcopy(duplicate_count_template) for _ in range(3))
        skip_af_index = list()
        af_runtimes = [0, 0, 0]
        af_observations = [list(), list(), list()]
        initial_sample_mean = np.mean(self.__valid_observations)
        while self.fevals < max_fevals:
            time_start = time.perf_counter_ns()
            # the first acquisition function is never skipped, so that should be the best for the endgame (EI)
            aqfs = self.multi_afs
            predictions, _, std = self.predict_list(self.unvisited_cache)
            hyperparam = self.contextual_variance(std)
            if self.__visited_num >= self.searchspace_size:
                raise ValueError(self.error_message_searchspace_fully_observed)
            time_predictions = time.perf_counter_ns()
            actual_candidate_params = list()
            actual_candidate_indices = list()
            actual_candidate_af_indices = list()
            duplicate_candidate_af_indices = list()
            duplicate_candidate_original_af_indices = list()
            for af_index, af in enumerate(aqfs):
                if af_index in skip_af_index:
                    continue
                if self.__visited_num >= self.searchspace_size or self.fevals >= max_fevals:
                    break
                timer_start = time.perf_counter()
                list_of_acquisition_values = af(predictions, hyperparam)
                best_af = self.argopt(list_of_acquisition_values)
                time_taken = time.perf_counter() - timer_start
                af_runtimes[af_index] += time_taken
                is_duplicate = best_af in actual_candidate_indices
                if not is_duplicate:
                    candidate_params = self.unvisited_cache[best_af]
                    actual_candidate_params.append(candidate_params)
                    actual_candidate_indices.append(best_af)
                    actual_candidate_af_indices.append(af_index)
                # register whether the AF suggested a duplicate candidate
                duplicate_candidate_af_count[af_index].pop(0)
                duplicate_candidate_af_count[af_index].append(1 if is_duplicate else 0)
                if is_duplicate:
                    # find the index of the AF that first registered the duplicate
                    original_duplicate_af_index = actual_candidate_af_indices[actual_candidate_indices.index(best_af)]
                    # register that AF as duplicate as well
                    duplicate_candidate_af_count[original_duplicate_af_index][-1] = 1
                    duplicate_candidate_af_indices.append(af_index)
                    duplicate_candidate_original_af_indices.append(original_duplicate_af_index)
            time_afs = time.perf_counter_ns()
            # evaluate the non-duplicate candidates
            for index, af_index in enumerate(actual_candidate_af_indices):
                candidate_params = actual_candidate_params[index]
                candidate_index = self.find_param_config_index(candidate_params)
                observation = self.evaluate_objective_function(candidate_params)
                self.update_after_evaluation(observation, candidate_index, candidate_params)
                if observation != self.invalid_value:
                    # we use the registered observations for maximization of the discounted reward
                    reg_observation = observation if self.opt_direction == 'min' else -1 * observation
                    af_observations[actual_candidate_af_indices[index]].append(reg_observation)
                else:
                    reg_invalid_observation = initial_sample_mean if self.opt_direction == 'min' else -1 * initial_sample_mean
                    af_observations[actual_candidate_af_indices[index]].append(reg_invalid_observation)
            for index, af_index in enumerate(duplicate_candidate_af_indices):
                original_observation = af_observations[duplicate_candidate_original_af_indices[index]][-1]
                af_observations[af_index].append(original_observation)
            self.fit_observations_to_model()
            time_eval = time.perf_counter_ns()
            # assert that all observation lists of non-skipped acquisition functions are of the same length
            non_skipped_af_indices = list(af_index for af_index, _ in enumerate(aqfs) if af_index not in skip_af_index)
            assert all(len(af_observations[non_skipped_af_indices[0]]) == len(af_observations[af_index]) for af_index in non_skipped_af_indices)
            # find the AFs elligble for being skipped
            candidates_for_skip = list()
            for af_index, count in enumerate(duplicate_candidate_af_count):
                if sum(count) >= skip_if_duplicate_n_times and af_index not in skip_af_index:
                    candidates_for_skip.append(af_index)
            # do not skip the AF with the lowest runtime
            if len(candidates_for_skip) > 1:
                candidates_for_skip_discounted = list(
                    sum(list(obs * discount_factor**(len(observations) - 1 - i) for i, obs in enumerate(observations)))
                    for af_index, observations in enumerate(af_observations) if af_index in candidates_for_skip)
                af_not_to_skip = candidates_for_skip[np.argmin(candidates_for_skip_discounted)]
                for af_index in candidates_for_skip:
                    if af_index == af_not_to_skip:
                        # do not skip the AF with the lowest runtime and give it a clean slate
                        duplicate_candidate_af_count[af_index] = deepcopy(duplicate_count_template)
                        continue
                    skip_af_index.append(af_index)
                    if len(skip_af_index) >= len(aqfs):
                        raise ValueError("There are no acquisition functions left! This should not happen...")
            time_af_selection = time.perf_counter_ns()

            # printing timings
            if self.log_timings:
                time_taken_predictions = round(time_predictions - time_start, 3) / 1000
                time_taken_afs = round(time_afs - time_predictions, 3) / 1000
                time_taken_eval = round(time_eval - time_afs, 3) / 1000
                time_taken_af_selection = round(time_af_selection - time_eval, 3) / 1000
                time_taken_total = round(time_af_selection - time_start, 3) / 1000
                print(
                    f"({self.fevals}/{max_fevals}) Total time: {time_taken_total} | Predictions: {time_taken_predictions} | AFs: {time_taken_afs} | Eval: {time_taken_eval} | AF selection: {time_taken_af_selection}",
                    flush=True)
        return self.results

    def __optimize_multi_advanced(self, max_fevals, increase_precision=False):
        """ Optimize with a portfolio of multiple acquisition functions. Predictions are only taken once, unless increase_precision is true. Skips AFs if they are consistently worse than the mean of discounted observations, promotes AFs if they are consistently better than this mean. """
        if self.opt_direction != 'min':
            raise ValueError(f"Optimization direction must be minimization ('min'), is {self.opt_direction}")
        aqfs = self.multi_afs
        discount_factor = self.multi_afs_discount_factor
        required_improvement_factor = self.multi_afs_required_improvement_factor
        required_improvement_worse = 1 + required_improvement_factor
        required_improvement_better = 1 - required_improvement_factor
        min_required_count = self.af_params['skip_duplicate_after']
        skip_af_index = list()
        single_af = len(aqfs) <= len(skip_af_index) + 1
        af_observations = [list(), list(), list()]
        af_performs_worse_count = [0, 0, 0]
        af_performs_better_count = [0, 0, 0]
        while self.fevals < max_fevals:
            if single_af:
                return self.__optimize(max_fevals)
            if self.__visited_num >= self.searchspace_size:
                raise ValueError(self.error_message_searchspace_fully_observed)
            observations_median = np.median(self.__valid_observations)
            if increase_precision is False:
                predictions, _, std = self.predict_list(self.unvisited_cache)
                hyperparam = self.contextual_variance(std)
            for af_index, af in enumerate(aqfs):
                if af_index in skip_af_index:
                    continue
                if self.__visited_num >= self.searchspace_size or self.fevals >= max_fevals:
                    break
                if increase_precision is True:
                    predictions, _, std = self.predict_list(self.unvisited_cache)
                    hyperparam = self.contextual_variance(std)
                list_of_acquisition_values = af(predictions, hyperparam)
                best_af = self.argopt(list_of_acquisition_values)
                del predictions[best_af]    # to avoid going out of bounds
                candidate_params = self.unvisited_cache[best_af]
                candidate_index = self.find_param_config_index(candidate_params)
                observation = self.evaluate_objective_function(candidate_params)
                self.update_after_evaluation(observation, candidate_index, candidate_params)
                if increase_precision is True:
                    self.fit_observations_to_model()
                # we use the registered observations for maximization of the discounted reward
                if observation != self.invalid_value:
                    reg_observation = observation if self.opt_direction == 'min' else -1 * observation
                    af_observations[af_index].append(reg_observation)
                else:
                    # if the observation is invalid, use the median of all valid observations to avoid skewing the discounted observations
                    reg_invalid_observation = observations_median if self.opt_direction == 'min' else -1 * observations_median
                    af_observations[af_index].append(reg_invalid_observation)
            if increase_precision is False:
                self.fit_observations_to_model()

            # calculate the mean of discounted observations over the remaining acquisition functions
            discounted_obs = list(
                sum(list(obs * discount_factor**(len(observations) - 1 - i) for i, obs in enumerate(observations))) for observations in af_observations)
            disc_obs_mean = np.mean(list(discounted_obs[af_index] for af_index, _ in enumerate(aqfs) if af_index not in skip_af_index))

            # register which AFs perform more than 10% better than average and which more than 10% worse than average
            for af_index, discounted_observation in enumerate(discounted_obs):
                if discounted_observation > disc_obs_mean * required_improvement_worse:
                    af_performs_worse_count[af_index] += 1
                elif discounted_observation < disc_obs_mean * required_improvement_better:
                    af_performs_better_count[af_index] += 1

            # find the worst AF, discounted observations is leading for a draw
            worst_count = max(list(count for af_index, count in enumerate(af_performs_worse_count) if af_index not in skip_af_index))
            af_index_worst = -1
            if worst_count >= min_required_count:
                for af_index, count in enumerate(af_performs_worse_count):
                    if af_index not in skip_af_index and count == worst_count and (af_index_worst == -1
                                                                                   or discounted_obs[af_index] > discounted_obs[af_index_worst]):
                        af_index_worst = af_index

            # skip the worst AF
            if af_index_worst > -1:
                skip_af_index.append(af_index_worst)
                # reset the counts to even the playing field for the remaining AFs
                af_performs_worse_count = [0, 0, 0]
                af_performs_better_count = [0, 0, 0]
                # if there is only one AF left, register as single AF
                if len(aqfs) <= len(skip_af_index) + 1:
                    single_af = True
                    af_indices_left = list(af_index for af_index, _ in enumerate(aqfs) if af_index not in skip_af_index)
                    assert len(af_indices_left) == 1
                    self.__af = aqfs[af_indices_left[0]]
            else:
                # find the best AF, discounted observations is leading for a draw
                best_count = max(list(count for af_index, count in enumerate(af_performs_better_count) if af_index not in skip_af_index))
                af_index_best = -1
                if best_count >= min_required_count:
                    for af_index, count in enumerate(af_performs_better_count):
                        if af_index not in skip_af_index and count == best_count and (af_index_best == -1
                                                                                      or discounted_obs[af_index] < discounted_obs[af_index_best]):
                            af_index_best = af_index
                # make the best AF single
                if af_index_best > -1:
                    single_af = True
                    self.__af = aqfs[af_index_best]

        return self.results

    def __optimize_multi_fast(self, max_fevals):
        """ Optimize with a portfolio of multiple acquisition functions. Predictions are only taken once. """
        while self.fevals < max_fevals:
            aqfs = self.multi_afs
            # if we take the prediction only once, we want to go from most exploiting to most exploring, because the more exploiting an AF is, the more it relies on non-stale information from the model
            predictions, _, std = self.predict_list(self.unvisited_cache)
            hyperparam = self.contextual_variance(std)
            if self.__visited_num >= self.searchspace_size:
                raise ValueError(self.error_message_searchspace_fully_observed)
            for af in aqfs:
                if self.__visited_num >= self.searchspace_size or self.fevals >= max_fevals:
                    break
                list_of_acquisition_values = af(predictions, hyperparam)
                best_af = self.argopt(list_of_acquisition_values)
                del predictions[best_af]    # to avoid going out of bounds
                candidate_params = self.unvisited_cache[best_af]
                candidate_index = self.find_param_config_index(candidate_params)
                observation = self.evaluate_objective_function(candidate_params)
                self.update_after_evaluation(observation, candidate_index, candidate_params)
            self.fit_observations_to_model()
        return self.results

    def af_random(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function returning a randomly shuffled list for comparison """
        list_random = range(len(self.unvisited_cache))
        shuffle(list_random)
        return list_random

    def af_probability_of_improvement(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function Probability of Improvement (PI) """

        # prefetch required data
        if predictions is None:
            predictions, _, _ = self.predict_list(self.unvisited_cache)
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        fplus = self.current_optimum - hyperparam

        # precompute difference of improvement
        list_diff_improvement = list(-((fplus - x_mu) / (x_std + 1E-9)) for (x_mu, x_std) in predictions)

        # compute probability of improvement with CDF in bulk
        list_prob_improvement = norm.cdf(list_diff_improvement)

        return list_prob_improvement

    def af_expected_improvement(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function Expected Improvement (EI) """

        # prefetch required data
        if predictions is None:
            predictions, _, _ = self.predict_list(self.unvisited_cache)
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        fplus = self.current_optimum - hyperparam

        # precompute difference of improvement, CDF and PDF in bulk
        list_diff_improvement = list((fplus - x_mu) / (x_std + 1E-9) for (x_mu, x_std) in predictions)
        list_cdf = norm.cdf(list_diff_improvement)
        list_pdf = norm.pdf(list_diff_improvement)

        # specify AF calculation
        def exp_improvement(index) -> float:
            x_mu, x_std = predictions[index]
            ei = (fplus - x_mu) * list_cdf[index] + x_std * list_pdf[index]
            return -ei

        # calculate AF
        list_exp_improvement = list(map(exp_improvement, range(len(predictions))))
        return list_exp_improvement

    def af_lower_confidence_bound(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function Lower Confidence Bound (LCB) """

        # prefetch required data
        if predictions is None:
            predictions, _, _ = self.predict_list(self.unvisited_cache)
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        beta = hyperparam

        # compute LCB in bulk
        list_lower_confidence_bound = list(x_mu - beta * x_std for (x_mu, x_std) in predictions)
        return list_lower_confidence_bound

    def af_lower_confidence_bound_srinivas(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function Lower Confidence Bound (UCB-S) after Srinivas, 2010 / Brochu, 2010 """

        # prefetch required data
        if predictions is None:
            predictions, _, _ = self.predict_list(self.unvisited_cache)
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']

        # precompute beta parameter
        zeta = self.af_params['zeta']
        t = self.fevals
        d = self.num_dimensions
        delta = hyperparam
        beta = np.sqrt(zeta * (2 * np.log((t**(d / 2. + 2)) * (np.pi**2) / (3. * delta))))

        # compute UCB in bulk
        list_lower_confidence_bound = list(x_mu - beta * x_std for (x_mu, x_std) in predictions)
        return list_lower_confidence_bound

    def visualize_after_opt(self):
        """ Visualize the model after the optimization """
        print(self.__model.kernel_.get_params())
        print(self.__model.log_marginal_likelihood())
        import matplotlib.pyplot as plt
        _, mu, std = self.predict_list(self.searchspace)
        brute_force_observations = list()
        for param_config in self.searchspace:
            obs = minimize._cost_func(param_config, self.kernel_options, self.tuning_options, self.runner, self.results)
            if obs == self.invalid_value:
                obs = None
            brute_force_observations.append(obs)
        x_axis = range(len(mu))
        plt.fill_between(x_axis, mu - std, mu + std, alpha=0.2, antialiased=True)
        plt.plot(x_axis, mu, label="predictions", linestyle=' ', marker='.')
        plt.plot(x_axis, brute_force_observations, label="actual", linestyle=' ', marker='.')
        plt.legend()
        plt.show()
