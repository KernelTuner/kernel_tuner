""" A simple genetic algorithm for parameter search """
from __future__ import print_function

from collections import OrderedDict

from numpy.core.fromnumeric import mean

# BO2 imports

try:
    import numpy as np
    from typing import Tuple
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
    from skopt.sampler import Lhs
    from random import randint, seed, shuffle
    from random import sample as randsample
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

supported_cov_kernels = ["constantrbf", "rbf", "matern32", "matern52"]
supported_methods = ["poi", "ei", "ucb", "ucb-srinivas", "ts"]
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
    multi_opt = tuning_options.strategy_options.get("optimizemulti", False)
    cov_kernel = tuning_options.strategy_options.get("covariancekernel", "matern52")
    acq = tuning_options.strategy_options.get("method", "ei")
    acq_params = tuning_options.strategy_options.get("methodparams", {
        'explorationfactor': 'CV',
        'zeta': 1,
    })
    acq_num = tuning_options.strategy_options.get("numacquisition", 1)
    init_points = tuning_options.strategy_options.get("popsize", 20)
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
    bo = BayesianOptimization(parameter_space, kernel_options, tuning_options, normalize_dict, denormalize_dict, runner, init_points,
                              cov_kernel_name=cov_kernel, acquisition_function=acq, acquisition_function_num=acq_num, acq_func_params=acq_params,
                              sampling_method=sampling_method, sampling_crit=sampling_crit, sampling_iter=sampling_iter)
    # optimize
    time_opt = time.perf_counter()
    if not multi_opt:
        results = bo.optimize(max_fevals)
    else:
        results = bo.optimize_multi(max_fevals)
    time_end = time.perf_counter()
    # print("Total: {} | Init: {} | Opt: {}".format(round(time_end - time_init, 3), round(time_opt - time_init, 3), round(time_end - time_opt, 3)))
    return results, runner.dev.get_environment()


class BayesianOptimization():

    def __init__(self, searchspace: list, kernel_options: dict, tuning_options: dict, normalize_dict: dict, denormalize_dict: dict, runner,
                 num_initial_samples: int, opt_direction='min', cov_kernel_name='default', acquisition_function='ei', acquisition_function_num=1,
                 acq_func_params=None, sampling_method='lhs', sampling_crit=None, sampling_iter=1000):
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
            self.__af = self.af_probability_of_improvement
        elif acquisition_function == 'ei':
            self.__af = self.af_expected_improvement
        elif acquisition_function == 'ucb':
            self.__af = self.af_upper_confidence_bound
        elif acquisition_function == 'ucb-srinivas':
            self.__af = self.af_upper_confidence_bound_srinivas
        elif acquisition_function == 'ts':
            self.__af = self.af_thompson_sampling
        elif acquisition_function == 'random':
            self.__af = self.af_random
        else:
            raise ValueError("Acquisition function must be one of {}, is {}".format(supported_methods, acquisition_function))

        # set kernel and Gaussian process
        if cov_kernel_name == "constantrbf":
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
        elif cov_kernel_name == "rbf":
            kernel = RBF(1.0, length_scale_bounds="fixed")
        elif cov_kernel_name == "matern32":
            kernel = Matern(length_scale=1.0, nu=1.5, length_scale_bounds="fixed")
        elif cov_kernel_name == "matern52":
            kernel = Matern(length_scale=1.0, nu=2.5, length_scale_bounds="fixed")
        else:
            raise ValueError("Acquisition function must be one of {}, is {}".format(supported_cov_kernels, cov_kernel_name))
        self.__model = GaussianProcessRegressor(kernel=1.0 * kernel, normalize_y=False)

        # set remaining values
        self.results = []
        self.__searchspace = searchspace
        self.searchspace_size = len(self.searchspace)
        self.num_dimensions = len(self.dimensions())
        self.__current_optimum = self.worst_value
        self.__visited_num = 0
        self.__visited_valid_num = 0
        self.__visited_searchspace_indices = [False] * self.searchspace_size
        self.__observations = [np.NaN] * self.searchspace_size
        self.__valid_observation_indices = [False] * self.searchspace_size
        self.__valid_params = list()
        self.__valid_observations = list()
        self.unvisited_cache = self.unvisited()
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

    def get_current_optimum(self) -> Tuple[list, float]:
        """ Return the current optimum parameter configuration and its value """
        # TODO deprecated, no longer valid
        params, observations = self.valid_params_observations()
        if len(params) == 0:
            raise ValueError("No valid observation found, so no optimum either")
        index = self.argopt(observations)
        return params[index], observations[index]

    def valid_params_observations(self) -> Tuple[list, list]:
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
            self.__visited_valid_num += 1
            self.__valid_params.append(param_config)
            self.__valid_observations.append(observation)
            if self.is_better_than(observation, self.current_optimum):
                self.current_optimum = observation

    def predict(self, x) -> Tuple[float, float]:
        """ Returns a mean and standard deviation predicted by the surrogate model for the parameter configuration """
        return self.__model.predict([x], return_std=True)

    def predict_list(self, lst: list) -> Tuple[list, list]:
        """ Returns a list of means and standard deviations predicted by the surrogate model for the parameter configurations, and a list of standard deviations """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, std = self.__model.predict(lst, return_std=True)
            return list(zip(mu, std)), std

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

    def draw_random_sample(self) -> Tuple[list, int]:
        """ Draw a random sample from the unvisited parameter configurations """
        index = randint(0, len(self.unvisited_cache) - 1)
        param_config = self.unvisited_cache[index]
        actual_index = self.find_param_config_index(param_config)
        return param_config, actual_index

    def draw_latin_hypercube_samples(self, num_samples: int) -> list:
        """ Draws an LHS-distributed sample from the search space """
        if self.searchspace_size < num_samples:
            raise ValueError("Can't sample more than the size of the search space")
        # TODO test which is the best, maximin or other criterion is probably best but takes longer due to iterations
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
        _, std = self.predict_list(self.unvisited_cache)
        self.initial_sample_mean = np.mean(self.__valid_observations)
        self.initial_sample_std = np.mean(std)
        self.cv_maximum = self.initial_sample_std

    def contextual_variance(self, std: list) -> float:
        """ Contextual improvement to decide explore / exploit, based on CI proposed by (Jasrasaria, 2018) """
        if not self.af_params['explorationfactor'] == 'CV':
            return None
        if self.opt_direction == 'min':
            if self.current_optimum < 0:
                # TODO doesn't work well for minimization beyond 0, should that even be a thing?
                return abs(np.mean(std) / self.current_optimum)
            cv = np.mean(std) / (self.initial_sample_mean / self.current_optimum)
            cv_normalized = cv / self.cv_maximum
            iteration_weight = np.sqrt(1 / self.__visited_num)
            return cv_normalized * iteration_weight
        return np.mean(std) / self.current_optimum

    def optimize(self, max_function_evaluations):
        """ Find the next best candidate configuration(s), evaluate those and update the model accordingly """
        while self.__visited_num < max_function_evaluations:
            # time_af = time.perf_counter()
            if self.__visited_num >= self.searchspace_size:
                raise ValueError("The search space has been fully observed")
            predictions, std = self.predict_list(self.unvisited_cache)
            hyperparam = self.contextual_variance(std)
            list_of_acquisition_values = self.__af(predictions, hyperparam)
            # afterwards select the best AF value
            # time_select = time.perf_counter()
            if self.af_num == 1:
                best_af = self.argopt(list_of_acquisition_values)
                candidate_params = self.unvisited_cache[best_af]
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
        return self.results

    def optimize_multi(self, max_function_evaluations):
        """ Optimize with a portfolio of multiple acquisition functions """
        # TODO perhaps optimize by filtering similar candidates, in that case wait with update_after_evaluation to check if the same candidate is suggested
        while self.__visited_num < max_function_evaluations:
            aqfs = [self.af_expected_improvement, self.af_upper_confidence_bound, self.af_probability_of_improvement]
            predictions, std = self.predict_list(self.unvisited_cache)
            hyperparam = self.contextual_variance(std)
            if self.__visited_num >= self.searchspace_size:
                raise ValueError("The search space has been fully observed")
            for af in aqfs:
                if self.__visited_num >= self.searchspace_size or self.__visited_num >= max_function_evaluations:
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
            predictions, _ = self.predict_list(self.unvisited_cache)
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        fplus = self.current_optimum + hyperparam

        # precompute difference of improvement
        list_diff_improvement = list((x_mu - fplus) / (x_std + 1E-9) for (x_mu, x_std) in predictions)

        # compute probability of improvement with CDF in bulk
        list_prob_improvement = norm.cdf(list_diff_improvement)

        return list_prob_improvement

    def af_expected_improvement(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function Expected Improvement (EI) """

        # prefetch required data
        # time_prefetch = time.perf_counter()
        if predictions is None:
            predictions, _ = self.predict_list(self.unvisited_cache)
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        fplus = self.current_optimum + hyperparam

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

    def af_upper_confidence_bound(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function Upper Confidence Bound (UCB) """

        # prefetch required data
        if predictions is None:
            predictions, _ = self.predict_list(self.unvisited_cache)
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        beta = hyperparam

        # compute UCB in bulk
        list_upper_confidence_bound = list(x_mu + beta * x_std for (x_mu, x_std) in predictions)

        return list_upper_confidence_bound

    def af_upper_confidence_bound_srinivas(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function Upper Confidence Bound (UCB-S) after Srinivas, 2010 / Brochu, 2010 """

        # prefetch required data
        if predictions is None:
            predictions, _ = self.predict_list(self.unvisited_cache)
        if hyperparam is None:
            hyperparam = 0.1

        # precompute beta parameter
        zeta = self.af_params['zeta']
        t = self.__visited_num
        d = self.num_dimensions
        delta = hyperparam
        beta = np.sqrt(zeta * (2 * np.log((t**(d / 2. + 2)) * (np.pi**2) / (3. * delta))))

        # compute UCB in bulk
        list_upper_confidence_bound = list(x_mu + beta * x_std for (x_mu, x_std) in predictions)
        return list_upper_confidence_bound

    def af_thompson_sampling(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function Thompson Sampling (TS) """

        # prefetch required data
        # time_start = time.perf_counter()
        random_indices = randsample(range(0, len(self.unvisited_cache) - 1), 400)
        random_samples = [self.unvisited_cache[i] for i in random_indices]
        samples = self.__model.sample_y(random_samples, 1)
        # print(round(time.perf_counter() - time_start, 3))
        # print(samples)
        transposed = samples.T[0]
        # print(transposed)

        return transposed
