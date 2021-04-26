""" A simple genetic algorithm for parameter search """
from __future__ import print_function

from collections import OrderedDict

# BO2 imports

try:
    import numpy as np
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    from skopt.sampler import Lhs
    from copy import deepcopy
    from random import randint, seed, shuffle
    from random import uniform as randuni
    import itertools
    import warnings
    import time    # for time.perf_counter()
    import multiprocessing    # for multi-threaded AF calculation
    bayes_opt_present = True
except Exception:
    bayes_opt_present = False

from kernel_tuner.strategies import minimize
from kernel_tuner import util

supported_methods = ["poi", "ei", "ucb"]
supported_sampling_methods = ["random", "lhs"]


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
        process.
    :type tuning_options: kernel_tuner.interface.Options

    :returns: A list of dictionaries for executed kernel configurations and their
        execution times. And a dictionary that contains a information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    if not bayes_opt_present:
        raise ImportError("Error: optional dependencies for Bayesian Optimization not installed")

    def generate_normalized_param_dicts(tune_params: dict, eps: float) -> dict:
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
        keys = list(tune_params.keys())
        param_space_normalized = list(tuple(normalized[keys[i]][v] for i, v in enumerate(params)) for params in param_space)
        return param_space_normalized

    # get strategy options or defaults
    acq = tuning_options.strategy_options.get("method", "ei")
    acq_num = tuning_options.strategy_options.get("numacquisition", 1)
    init_points = tuning_options.strategy_options.get("popsize", 20)
    max_iter = tuning_options.strategy_options.get("maxiter", 100)
    max_fevals = tuning_options.strategy_options.get("max_fevals", 100)
    sampling_method = tuning_options.strategy_options.get("samplingmethod", "lhs")
    sampling_crit = tuning_options.strategy_options.get("samplingcriterion", None)
    sampling_iter = tuning_options.strategy_options.get("samplingiterations", 1000)

    # epsilon for scaling should be the evenly spaced distance between the largest set of parameter options in an interval [0,1]
    tune_params = tuning_options.tune_params
    tuning_options["scaling"] = True
    _, _, eps = minimize.get_bounds_x0_eps(tuning_options)

    # compute cartesian product of all tunable parameters
    parameter_space = itertools.product(*tune_params.values())
    # check for search space restrictions
    if tuning_options.restrictions is not None:
        parameter_space = filter(lambda p: util.check_restrictions(tuning_options.restrictions, p, tune_params.keys(), tuning_options.verbose), parameter_space)
    parameter_space = list(parameter_space)
    # normalize search space to [0,1]
    normalize_dict, denormalize_dict = generate_normalized_param_dicts(tune_params, eps)
    parameter_space = normalize_parameter_space(parameter_space, tune_params, normalize_dict)

    # initialize
    time_init = time.perf_counter()
    bo = BayesianOptimization(parameter_space, kernel_options, tuning_options, normalize_dict, denormalize_dict, runner, init_points, acquisition_function=acq,
                              acquisition_function_num=acq_num, sampling_method=sampling_method, sampling_crit=sampling_crit, sampling_iter=sampling_iter)
    # optimize
    time_opt = time.perf_counter()
    results = bo.optimize(max_iter, max_fevals)
    time_end = time.perf_counter()
    print("Total: {} | Init: {} | Opt: {}".format(round(time_end - time_init, 3), round(time_opt - time_init, 3), round(time_end - time_opt, 3)))
    return results, runner.dev.get_environment()


class BayesianOptimization():

    def __init__(self, searchspace: list, kernel_options: dict, tuning_options: dict, normalize_dict: dict, denormalize_dict: dict, runner,
                 num_initial_samples: int, opt_direction='min', acquisition_function='ei', acquisition_function_num=1, acq_func_params=None,
                 sampling_method='lhs', sampling_crit=None, sampling_iter=1000):
        # set arguments
        self.kernel_options = kernel_options
        self.tuning_options = tuning_options
        self.tune_params = tuning_options.tune_params
        self.param_names = list(self.tune_params.keys())
        self.normalized_dict = normalize_dict
        self.denormalized_dict = denormalize_dict
        self.sampling_method = sampling_method
        self.sampling_crit = sampling_crit
        self.sampling_iter = sampling_iter
        self.runner = runner
        self.max_threads = runner.dev.max_threads
        self.num_initial_samples = num_initial_samples

        # set optimization constants
        self.invalid_value = 1e20
        self.opt_direction = opt_direction
        if opt_direction == 'min':
            self.worst_value = np.PINF
            self.argopt = np.argmin
            self.af_num_partition = acquisition_function_num
        elif opt_direction == 'max':
            self.worst_value = np.NINF
            self.argopt = np.argmax
            self.af_num_partition = -1 * acquisition_function_num
        else:
            raise ValueError("Invalid optimization direction '{}'".format(opt_direction))
        self.af_num_cutoff = slice(self.af_num_partition)

        # set acquisition function
        if acquisition_function_num < 1:
            raise ValueError("Invalid number of acquisition function top values")
        self.af_num = acquisition_function_num
        self.af_params = acq_func_params
        self.predicted_unvisited = None
        self.cached_af_list = None
        if acquisition_function == 'poi':
            if self.af_params is None:
                self.af_params = {
                    'explorationfactor': 0.01
                }
            self.__af = self.af_probability_of_improvement
        elif acquisition_function == 'ei':
            if self.af_params is None:
                self.af_params = {
                    'explorationfactor': 0.01
                }
            self.__af = self.af_expected_improvement
        elif acquisition_function == 'ucb':
            self.__af = self.af_random
        else:
            raise ValueError("Acquisition function must be one of {}, is {}".format(supported_methods, acquisition_function))

        # set remaining values
        self.results = []
        self.__searchspace = searchspace
        self.searchspace_size = len(self.searchspace)
        self.__current_optimum = self.worst_value
        self.__visited_num = 0
        self.__visited_searchspace_indices = [False] * self.searchspace_size
        self.__observations = [np.NaN] * self.searchspace_size
        self.__valid_observation_indices = [False] * self.searchspace_size
        self.__valid_params = list()
        self.__valid_observations = list()
        self.unvisited_cache = self.unvisited()
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        self.__model = GaussianProcessRegressor(kernel=kernel)
        self.initial_sample()

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

    def get_current_optimum(self) -> (list, float):
        """ Return the current optimum parameter configuration and its value """
        # TODO deprecated, no longer valid
        params, observations = self.valid_params_observations()
        if len(params) == 0:
            raise ValueError("No valid observation found, so no optimum either")
        index = self.argopt(observations)
        return params[index], observations[index]

    def valid_params_observations(self) -> (list, list):
        """ Returns a list of valid observations and their parameter configurations """
        # if you do this every iteration, better keep it as cache and update in update_after_evaluation
        params = list()
        observations = list()
        for index, valid in enumerate(self.__valid_observation_indices):
            if valid is True:
                params.append(self.searchspace[index])
                observations.append(self.observations[index])
        # params = list(self.searchspace[index] for index, valid in enumerate(self.__valid_observation_indices) if valid is True)
        # observations = list(self.observations[index] for index, valid in enumerate(self.__valid_observation_indices) if valid is True)
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

    def update_after_evaluation(self, observation: float, index: int, param_config: tuple):
        """ Adjust the visited and valid index records accordingly """
        validity = self.is_valid(observation)
        self.__visited_num += 1
        self.__observations[index] = observation
        self.__visited_searchspace_indices[index] = True
        del self.unvisited_cache[self.find_param_config_unvisited_index(param_config)]
        self.__valid_observation_indices[index] = validity
        if validity is True:
            self.__valid_params.append(param_config)
            self.__valid_observations.append(observation)
            if self.is_better_than(observation, self.current_optimum):
                self.current_optimum = observation

    def predict(self, x) -> (float, float):
        """ Returns a mean and standard deviation predicted by the surrogate model for the parameter configuration """
        return self.__model.predict([x], return_std=True)

    def predict_list(self, lst: list) -> list:
        """ Returns a list of means and standard deviations predicted by the surrogate model for the parameter configurations """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, std = self.__model.predict(lst, return_std=True)
            return list(zip(mu, std))

    def fit_observations_to_model(self):
        """ Update the model based on the current list of observations """
        self.__model.fit(self.__valid_params, self.__valid_observations)
        # TODO this seems only marginally faster: self.predictions_cache = self.predict_list(self.unvisited_cache)

    def evaluate_objective_function(self, param_config: tuple) -> float:
        """ Evaluates the objective function """
        if not util.config_valid(self.denormalize_param_config(param_config), self.tuning_options, self.max_threads):
            return self.invalid_value
        return minimize._cost_func(param_config, self.kernel_options, self.tuning_options, self.runner, self.results)

    def dimensions(self) -> list:
        """ List of parameter values per parameter """
        return self.tune_params.values()

    def draw_random_sample(self) -> (list, int):
        """ Draw a random sample from the unvisited parameter configurations """
        index = randint(0, len(self.unvisited_cache) - 1)
        param_config = self.unvisited_cache[index]
        actual_index = self.find_param_config_index(param_config)
        return param_config, actual_index

    def draw_latin_hypercube_samples(self, num_samples: int) -> list:
        """ Draws an LHS-distributed sample from the search space """
        if self.searchspace_size < num_samples:
            raise ValueError("Can't sample more than the size of the search space")
        # TODO test which is the best, maximin or other criterion is probably best but takes a lot of time due to iterations
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
            raise ValueError("Sampling method must be one of {}, is {}".format(supported_sampling_methods, self.sampling_method))
        collected_samples = 0
        for params, index in samples:
            observation = self.evaluate_objective_function(params)
            self.update_after_evaluation(observation, index, params)
            if self.is_valid(observation):
                collected_samples += 1
        while collected_samples < self.num_initial_samples:
            params, index = self.draw_random_sample()
            observation = self.evaluate_objective_function(params)
            self.update_after_evaluation(observation, index, params)
            # check for validity to avoid having no actual initial samples
            if self.is_valid(observation):
                collected_samples += 1
        self.fit_observations_to_model()

    def get_candidates(self) -> list:
        """ Get the next candidate observation """
        if self.__visited_num >= self.searchspace_size:
            raise ValueError("The search space has been fully observed")
        return self.__af()

    def do_next(self):
        """ Find the next best candidate configuration(s), evaluate those and update the model accordingly """
        # time_af = time.perf_counter()
        list_of_acquisition_values = self.get_candidates()
        # afterwards select the best AF value
        # time_select = time.perf_counter()
        if self.af_num == 1:
            highest_af = self.argopt(list_of_acquisition_values)
            candidate_params = self.unvisited_cache[highest_af]
            candidate_index = self.find_param_config_index(candidate_params)
            observation = self.evaluate_objective_function(candidate_params)
            self.update_after_evaluation(observation, candidate_index, candidate_params)
        else:
            # if we need N candidates per AF execution, we take the top N and evaluate those (if they are not too close to each other?)
            top = np.argpartition(list_of_acquisition_values, self.af_num_partition)[self.af_num_cutoff]
            for top_index in top:
                candidate_params = self.unvisited_cache[top_index]
                candidate_index = self.find_param_config_index(candidate_params)
                observation = self.evaluate_objective_function(candidate_params)
                self.update_after_evaluation(observation, candidate_index, candidate_params)
        # time_fit = time.perf_counter()
        self.fit_observations_to_model()
        # time_end = time.perf_counter()
        # print("Total {} s | AF {} | Select {} | Fit {}".format(
        #     round(time_end - time_af, 3),
        #     round(time_select - time_af, 3),
        #     round(time_fit - time_select, 3),
        #     round(time_end - time_fit, 3),
        # ))

    def optimize(self, max_iterations, max_function_evaluations):
        iterations = 0
        while iterations < max_iterations and self.__visited_num < max_function_evaluations:
            self.do_next()
            iterations += 1
        return self.results

    def af_random(self) -> list:
        """ Acquisition function returning a randomly shuffled list for comparison """
        list_random = range(len(self.unvisited_cache))
        shuffle(list_random)
        return list_random

    def af_probability_of_improvement(self) -> list:
        """ Acquisition function Probability of Improvement (PI) """

        # prefetch required data
        predictions = self.predict_list(self.unvisited_cache)
        fplus = self.current_optimum + self.af_params['explorationfactor']

        # precompute difference of improvement
        list_diff_improvement = list((x_mu - fplus) / (x_std + 1E-9) for (x_mu, x_std) in predictions)

        # compute probability of improvement with CDF in bulk
        list_prob_improvement = norm.cdf(list_diff_improvement)

        return list_prob_improvement

    def af_expected_improvement(self) -> list:
        """ Acquisition function Expected Improvement (EI) """

        # prefetch required data
        # time_prefetch = time.perf_counter()
        predictions = self.predict_list(self.unvisited_cache)
        fplus = self.current_optimum + self.af_params['explorationfactor']

        # precompute difference of improvement, CDF and PDF in bulk
        # time_precompute = time.perf_counter()
        list_diff_improvement = list((x_mu - fplus) / (x_std + 1E-9) for (x_mu, x_std) in predictions)
        list_cdf = norm.cdf(list_diff_improvement)
        list_pdf = norm.pdf(list_diff_improvement)

        # specify AF calculation
        # time_calc = time.perf_counter()

        def exp_improvement(index) -> float:
            x_mu, x_std = predictions[index]
            return (x_mu - fplus) * list_cdf[index] + x_std * list_pdf[index]

        # calculate AF
        list_exp_improvement = list(map(exp_improvement, range(len(predictions) - 1)))

        # time_end = time.perf_counter()
        # print("Total {} s | prefetch {} | precompute {} | calc {}".format(
        #     round(time_end - time_prefetch, 3),
        #     round(time_precompute - time_prefetch, 3),
        #     round(time_calc - time_precompute, 3),
        #     round(time_end - time_calc, 3),
        # ))
        return list_exp_improvement

    # def af_upper_confidence_bound(self) -> (list, int, list):
    #     """ Acquisition function Upper Confidence Bound (UCB) """
