""" Lean implementation of Bayesian Optimization with GPyTorch """
from copy import deepcopy
from typing import Any, Tuple
from random import randint, shuffle, choice
from math import ceil
import numpy as np
from numpy.lib.arraysetops import unique
from numpy.random import default_rng
import torch
import gpytorch

from kernel_tuner.util import get_valid_configs, config_valid
from kernel_tuner.strategies import minimize

supported_initial_sample_methods = ['lhs', 'index', 'random']
supported_methods = ['ei', 'poi', 'random']
supported_cov_kernels = ['matern', 'matern_scalekernel']
supported_likelihoods = ['Gaussian', 'GaussianPrior', 'FixedNoise']
supported_optimizers = ['LBFGS', 'Adam']


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

    # set CUDA availability
    use_cuda = False
    cuda_available = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda:0" if cuda_available else "cpu")
    if cuda_available:
        print(f"CUDA is available, device: {torch.cuda.get_device_name(device)}")


    # retrieve options with defaults
    options = tuning_options.strategy_options
    optimization_direction = options.get("optimization_direction", 'min')
    num_initial_samples = options.get("popsize", 20)
    max_fevals = options.get("max_fevals", 100)
    max_threads = runner.dev.max_threads
    if max_fevals < num_initial_samples:
        raise ValueError(f"Maximum number of function evaluations ({max_fevals}) can not be lower than the number of initial samples ({num_initial_samples}) ")

    # enabling scaling will unscale and snap inputs on evaluation, more efficient to keep unscale values in a lookup table
    tuning_options["snap"] = True
    tuning_options["scaling"] = False

    # prune the search space using restrictions
    parameter_space = get_valid_configs(tuning_options, max_threads)

    # limit max_fevals to max size of the parameter space
    max_fevals = min(len(parameter_space), max_fevals)

    # execute Bayesian Optimization
    BO = BayesianOptimization(parameter_space, kernel_options, tuning_options, runner, num_initial_samples, optimization_direction, device)
    # BO.visualize()
    all_results = BO.optimize(max_fevals)
    # BO.visualize()

    return all_results, runner.dev.get_environment()


class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, cov_kernel_name: str, cov_kernel_lengthscale: float):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if cov_kernel_name == 'matern':
            self.covar_module = gpytorch.kernels.MaternKernel(nu=cov_kernel_lengthscale)
        elif cov_kernel_name == 'matern_scalekernel':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=cov_kernel_lengthscale))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesianOptimization:

    def __init__(self, parameter_space: list, kernel_options, tuning_options, runner, num_initial_samples: int, optimization_direction: str, device: torch.device) -> None:
        self.animate = False    # TODO remove

        # set defaults
        self.num_initial_samples = num_initial_samples
        self.fevals = 0
        self.all_results = []
        self.unique_results = {}
        self.current_optimal_config = None

        # set Kernel Tuner data
        self.kernel_options = kernel_options
        self.tuning_options = tuning_options
        self.runner = runner
        self.max_threads = runner.dev.max_threads

        # get tuning options
        self.initial_sample_method = self.get_hyperparam("initialsamplemethod", "lhs", supported_initial_sample_methods)
        self.initial_sample_random_offset_factor = self.get_hyperparam("initialsamplerandomoffsetfactor", 0.1)
        self.initial_training_iter = self.get_hyperparam("initialtrainingiter", 50)
        self.training_iter = self.get_hyperparam("trainingiter", 3)
        self.cov_kernel_name = self.get_hyperparam("covariancekernel", "matern_scalekernel", supported_cov_kernels)
        self.cov_kernel_lengthscale = self.get_hyperparam("covariancelengthscale", 1.5)
        self.likelihood_name = self.get_hyperparam("likelihood", "Gaussian", supported_likelihoods)
        self.optimizer_name = self.get_hyperparam("optimizer", "Adam", supported_optimizers)
        self.optimizer_learningrate = self.get_hyperparam("optimizer_learningrate", 0.1)
        acquisition_function_name = self.get_hyperparam("method", "ei", supported_methods)
        af_params = self.get_hyperparam("methodparams", {})

        # set acquisition function options
        self.set_acquisition_function(acquisition_function_name)
        if 'explorationfactor' not in af_params:
            af_params['explorationfactor'] = 'CV'
        self.af_params = af_params

        # set Tensors
        # the unvisited_configs and valid_configs are to be used as boolean masks on the other tensors, more efficient than adding to / removing from tensors
        self.device = device
        self.out_device = torch.device("cpu")
        self.dtype = torch.double
        self.size = len(parameter_space)
        self.unvisited_configs = torch.ones(self.size, dtype=torch.bool).to(device)
        self.index_counter = torch.arange(self.size)
        self.valid_configs = torch.zeros(self.size, dtype=torch.bool).to(device)
        self.inital_sample_configs = torch.zeros(self.size, dtype=torch.bool).to(device)
        self.results = torch.zeros(self.size, dtype=self.dtype).to(device) * np.nan             # x (param configs) and y (results) must be the same type
        self.results_std = torch.ones(self.size, dtype=self.dtype).to(device) * 1e-3

        # transform non-numerical parameters to numerical, keep true_param_configs for evaluation function
        self.param_configs, self.tune_params = self.transform_nonnumerical_params(parameter_space)
        self.true_param_configs = parameter_space

        # set scaling
        self.scaled_input = True
        self.scaled_output = True
        if not self.scaled_input:
            self.param_configs_scaled = self.param_configs
        else:
            self.apply_scaling_to_inputs()

        # set optimization settings
        self.invalid_value = 1e20
        self.optimization_direction = optimization_direction
        if self.optimization_direction == 'min':
            self.is_better_than = lambda a, b: a < b
            self.inf_value = np.PINF
            self.opt = torch.min
            self.argopt = torch.argmin
        elif self.optimization_direction == 'max':
            self.is_better_than = lambda a, b: a > b
            self.inf_value = np.NINF
            self.opt = torch.max
            self.argopt = torch.argmax
        else:
            raise ValueError(f"Invalid optimization direction {self.optimization_direction}")

        # set the model
        self.current_optimum = self.inf_value
        self.hyperparams = {
            'loss': np.nan,
            'lengthscale': np.nan,
            'noise': np.nan,
        }
        self.initialize_model()

    @property
    def train_x(self):
        """ Get the valid parameter configurations """
        return self.param_configs_scaled[self.valid_configs].to(self.device)

    @property
    def train_y(self):
        """ Get the valid results """
        outputs = self.results[self.valid_configs]
        if self.scaled_output:
            # z-score, remove mean and make unit variance to scale it to N(0,1)
            # alternatively, first min-max the outputs between -1 and +1 and apply a Fisher transformation (np.arctanh)
            outputs = (outputs - outputs.mean()) / outputs.std()
        return outputs

    @property
    def train_y_err(self):
        """ Get the error on the valid results """
        std = self.results_std[self.valid_configs]
        if self.scaled_output and std.std() > 0.0:
            std = (std - std.mean()) / std.std()
        return std

    @property
    def test_x(self):
        """ Get the not yet visited parameter configurations """
        return self.param_configs_scaled[self.unvisited_configs].to(self.device)

    @property
    def test_x_unscaled(self):
        """ Get the unscaled, not yet visited parameter configurations """
        return self.param_configs[self.unvisited_configs]

    @property
    def invalid_x(self):
        """ Get the invalid parameter configurations by checking which visited configs are not valid (equivalent to checking which unvisited configs are valid) """
        invalid_mask = (self.unvisited_configs == self.valid_configs)
        return self.param_configs[invalid_mask]

    def true_param_config_index(self, target_index: int) -> int:
        """ The index required to get the true config param index when dealing with test_x """
        # get the index of the #index-th True (for example the 9th+1 True could be index 13 because there are 4 Falses in between)
        masked_counter = self.index_counter[self.unvisited_configs]
        return masked_counter[target_index]

    def true_param_config_indices(self, target_indices: torch.Tensor) -> torch.Tensor:
        """ Same as true_param_config_index, but for an array of targets instead. """
        masked_counter = self.index_counter[self.unvisited_configs]
        return masked_counter.index_select(0, target_indices)

    def initialize_model(self):
        """ Initialize the surrogate model """
        self.initial_sample()

        # create the model
        if self.likelihood_name == 'Gaussian':
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        elif self.likelihood_name == 'FixedNoise':
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=self.train_y_err.clamp(min=1.0e-4), learn_additional_noise=False)
        self.likelihood = self.likelihood.to(self.device)
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood, self.cov_kernel_name, self.cov_kernel_lengthscale)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        # LBFGS is probably better as Adam is only first-order
        if self.optimizer_name == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(model_parameters, lr=self.optimizer_learningrate)
        elif self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(model_parameters, lr=self.optimizer_learningrate)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.device)
        self.train_hyperparams(self.initial_training_iter)

    def initial_sample(self):
        """ Take an initial sample of the parameter space """
        list_param_config_indices = list()

        # generate a random offset from a normal distribution to add to the sample indices
        rng = default_rng()
        if self.initial_sample_random_offset_factor > 0.5:
            raise ValueError("Random offset factor should not be greater than 0.5 to avoid overlapping index offsets")
        random_offset_size = (self.size / self.num_initial_samples) * self.initial_sample_random_offset_factor
        random_offsets = np.round(rng.standard_normal(self.num_initial_samples) * random_offset_size)

        # first apply the initial sampling method
        if self.initial_sample_method == 'lhs':
            indices = self.get_lhs_samples(random_offsets)
            for param_config_index in indices.tolist():
                list_param_config_indices.append(param_config_index)
                self.evaluate_config(param_config_index)
        elif self.initial_sample_method == 'random':
            while self.fevals < self.num_initial_samples:
                param_config_index = randint(0, self.size - 1)
                if param_config_index in list_param_config_indices:
                    continue
                list_param_config_indices.append(param_config_index)
                self.evaluate_config(param_config_index)

        # then take index-spaced samples until all samples are valid
        while self.fevals < self.num_initial_samples:
            least_evaluated_region_index = self.get_middle_index_of_least_evaluated_region()
            param_config_index = min(max(int(least_evaluated_region_index + random_offsets[self.fevals].item()), 0), self.size-1)
            list_param_config_indices.append(param_config_index)
            self.evaluate_config(param_config_index)

        # set the current optimum, initial sample mean and initial sample std
        self.current_optimum = self.opt(self.train_y).item()
        self.initial_sample_mean = self.train_y.mean().item()
        self.initial_sample_std = None

        # save a boolean mask of the initial samples
        self.inital_sample_configs = self.valid_configs.detach().clone()

    def get_lhs_samples(self, random_offsets: np.ndarray) -> torch.Tensor:
        """ Get a centered Latin Hypercube Sample with a random offset """
        n_samples = self.num_initial_samples

        # first get the seperate parameter values to make possibly fictional distributed parameter configurations
        temp_param_configs = [[] for _ in range(n_samples)]
        for param_values in self.tune_params.values():
            l = len(param_values)

            # determine the interval and offset
            interval = l / n_samples
            offset = 0
            if l > n_samples:
                # take the difference between the last index and the end of the list, and the first index and the start of the list
                offset = ((l - 1 - interval * n_samples) - interval) / 2

            # assemble the parameter configurations
            for i in range(n_samples):
                index = ceil(offset + interval * (i + 1)) - 1
                temp_param_configs[i].append(param_values[index])

        # create a tensor of the possibly fictional parameter configurations
        param_configs = torch.tensor(list(tuple(param_config) for param_config in temp_param_configs), dtype=self.dtype).to(self.device)
        param_configs = param_configs.unique(dim=0) # remove duplicates
        n_samples_unique = len(param_configs)

        # get the indices of the parameter configurations
        num_params = len(self.param_configs[0])
        minimum_required_num_matching_params = round(num_params * 0.75)  # set the number of parameter matches allowed to be dropped before the search is stopped
        param_configs_indices = torch.full((n_samples_unique,), -1, dtype=torch.int)
        for selected_index, selected_param_config in enumerate(param_configs):
            # for each parameter configuration, count the number of matching parameters
            required_num_matching_params = num_params
            matching_params = torch.count_nonzero(self.param_configs == selected_param_config, -1)
            match_mask = (matching_params == required_num_matching_params)
            # if there is not at least one matching parameter configuration, lower the required number of matching parameters
            found_num_matching_param_configs = match_mask.count_nonzero()
            while found_num_matching_param_configs < 1 and required_num_matching_params > minimum_required_num_matching_params:
                required_num_matching_params -= 1
                match_mask = (matching_params == required_num_matching_params)
                found_num_matching_param_configs = match_mask.count_nonzero()

            # if more than one possible parameter configuration has been found, pick a random one
            if found_num_matching_param_configs > 1:
                index = choice(self.index_counter[match_mask])
            elif found_num_matching_param_configs == 1:
                index = self.index_counter[match_mask].item()
            else:
                # if no matching parameter configurations were found
                continue

            # set the selected index
            param_configs_indices[selected_index] = min(max(int(index + random_offsets[selected_index].item()), 0), self.size-1)

        # filter -1 indices and duplicates that occurred because of the random offset
        param_configs_indices = param_configs_indices[param_configs_indices >= 0]
        param_configs_indices = param_configs_indices.unique().type(torch.int)
        if len(param_configs_indices) < n_samples / 2:
            print(f"{n_samples - len(param_configs_indices)} out of the {n_samples} LHS samples were duplicates or -1.",
                  f"This might be because you have few initial samples ({n_samples}) relative to the number of parameters ({num_params}).",
                  "Perhaps try something other than LHS.")
        return param_configs_indices

    def get_middle_index_of_least_evaluated_region(self) -> int:
        """ Get the middle index of the region of parameter configurations that is the least visited """
        # This uses the largest distance between visited parameter configurations. That means it does not properly take the parameters into account, only the index of the parameter configurations, whereas LHS does.
        distance_tensor = torch.arange(self.size)

        # first get the indices that were visited (must be in ascending order)
        indices_visited = self.index_counter[~self.unvisited_configs]

        # then reset the range after the visited index
        for index_visited in indices_visited:
            distance_tensor[index_visited:] = torch.arange(self.size - index_visited)

        biggest_distance_index = distance_tensor.argmax()
        biggest_distance = distance_tensor[biggest_distance_index].item()
        middle_index = biggest_distance_index - round(biggest_distance / 2)
        # print(f"Max distance {biggest_distance}, index: {middle_index}, between: {biggest_distance_index-biggest_distance}-{biggest_distance_index}")
        return middle_index

    def find_nearest(self, value, array: torch.Tensor):
        """ Find the value nearest to the given value in the array """
        index = (torch.abs(array - value)).argmin()
        return array[index]

    def train_hyperparams(self, training_iter: int):
        """ Optimize the surrogate model hyperparameters iteratively """
        self.model.train()
        self.likelihood.train()

        def closure():
            self.optimizer.zero_grad()
            output = self.model(self.train_x)    # get model output
            try:
                loss = -self.mll(output, self.train_y)    # calculate loss and backprop gradients
                loss.backward()
                return loss
            except gpytorch.utils.errors.NotPSDError:
                print(f"WARNING - matrix not positive definite during training")

        loss = None
        for _ in range(training_iter):
            _loss = self.optimizer.step(closure)
            if _loss is not None:
                loss = _loss

        # set the hyperparams to the new values
        try:
            lengthscale = self.model.covar_module.lengthscale.item()
        except AttributeError:
            lengthscale = self.model.covar_module.base_kernel.lengthscale.item()
        self.hyperparams = {
            'loss': float(loss.item()) if loss is not None else np.nan,
            'lengthscale': float(lengthscale),
            'noise': float(self.model.likelihood.noise.mean().detach()),
        }

        # get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

    def optimize(self, max_fevals: int) -> Tuple[tuple, float]:
        """ Optimize the objective """
        predictions_tuple = None
        short_param_config_index = None
        last_invalid = False
        report_multiple_minima = round(self.size / 10)    # if more than 10% of the space is minima, print a warning
        use_contextual_variance = self.af_params['explorationfactor'] == 'CV'
        while self.fevals < max_fevals:
            if last_invalid:
                # TODO no need to get the predictions again as the predictions are unchanged, just set the invalid param config mean to the worst non-NAN value and the std to 0
                # predictions_tuple[0][short_param_config_index] = torch.nanmean(predictions_tuple[0])
                # predictions_tuple[1][short_param_config_index] = 0
                predictions_tuple = self.remove_from_predict_list(predictions_tuple, short_param_config_index)
            else:
                predictions_tuple = self.predict_list()
                if self.initial_sample_std is None:
                    self.initial_sample_std = predictions_tuple[1].mean().item()
            hyperparam = self.contextual_variance(predictions_tuple[0], predictions_tuple[1]) if use_contextual_variance else None
            acquisition_values = self.acquisition_function(predictions_tuple, hyperparam)
            short_param_config_index = self.argopt(acquisition_values)
            param_config_index = self.true_param_config_index(short_param_config_index)

            # if there are multiple minima in the acquisition function values, we want to take one from the least evaluated region
            min_acquisition_function_value = acquisition_values[short_param_config_index]
            indices_where_min = (acquisition_values <= min_acquisition_function_value).nonzero(as_tuple=True)[0]
            if len(indices_where_min) > 1:
                # first get the true index for the minima
                true_indices_where_min = self.true_param_config_indices(indices_where_min)
                # then get the index of the least evaluated region
                least_evaluated_region_index = self.get_middle_index_of_least_evaluated_region()
                # now find the minima closest to the least evaluated region
                param_config_index = self.find_nearest(least_evaluated_region_index, true_indices_where_min)
                short_param_config_index = -1    # invalidate the short_param_config_index because we bypassed it
                if len(indices_where_min) > report_multiple_minima:
                    print(
                        f"WARNING - after {self.fevals}/{max_fevals} fevals, there were multiple minima in the acquisition values ({len(indices_where_min)}), picking one based on the least evaluated region"
                    )

            # evaluate and register the result
            result = self.evaluate_config(param_config_index)
            if result == self.invalid_value and short_param_config_index > -1:
                # can't use last_invalid if there were multiple minima in the acquisition function values, because short_param_config_index will not be set
                last_invalid = True
            else:
                last_invalid = False
                self.model.set_train_data(self.train_x, self.train_y, strict=False)
                if self.training_iter > 0:
                    self.train_hyperparams(training_iter=self.training_iter)
                # set the current optimum
                self.current_optimum = self.opt(self.train_y).item()
            # print(f"Valid: {len(self.train_x)}, unvisited: {len(self.test_x)}, invalid: {len(self.invalid_x)}, last invalid: {last_invalid}")
            if self.animate:
                self.visualize()

        return self.all_results

    def objective_function(self, param_config: tuple) -> float:
        return minimize._cost_func(param_config, self.kernel_options, self.tuning_options, self.runner, self.all_results)

    def evaluate_config(self, param_config_index: int) -> float:
        """ Evaluates a parameter configuration, returns the time """
        param_config = self.true_param_configs[param_config_index]
        time = self.objective_function(param_config)
        self.register_result(time, param_config_index)
        self.update_unique_results()
        self.fevals = len(self.unique_results)
        return time

    def register_result(self, result: float, param_config_index: int):
        """ Registers the result to the Tensors and adds the hyperparameters to the results dict """
        # set the unvisited Tensors
        if self.unvisited_configs[param_config_index] == False:
            raise ValueError(f"The param config index {param_config_index} was already set to False!")
        self.unvisited_configs[param_config_index] = False

        # set the results Tensors
        last_result = self.all_results[-1]
        if result != self.invalid_value:
            self.valid_configs[param_config_index] = True
            self.results[param_config_index] = result
            assert last_result['time'] == result
            self.results_std[param_config_index] = np.std(last_result['times'])

        # add the current model parameters to the results dict
        if len(self.all_results) < 1:
            return
        for key, value in self.hyperparams.items():
            last_result[key] = value
        self.all_results[-1] = last_result

    def update_unique_results(self):
        """ Updates the unique results dictionary """
        record = self.all_results[-1]
        # make a unique string by taking every value in a result, if it already exists, it is overwritten
        self.unique_results.update({",".join([str(v) for k, v in record.items() if k in self.tuning_options.tune_params]): record["time"]})

    def predict_list(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns the means and standard deviations predicted by the surrogate model for the unvisited parameter configurations """
        with torch.no_grad(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(self.test_x))
            mu = observed_pred.mean
            std = observed_pred.variance.clamp(min=1e-9)    # TODO .sqrt() or not? looks like without is better
            return mu, std

    def remove_from_predict_list(self, p: Tuple[torch.Tensor, torch.Tensor], i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Remove an index from a tuple of predictions """
        return torch.cat([p[0][:i], p[0][i + 1:]]), torch.cat([p[1][:i], p[1][i + 1:]])

    def af_random(self, predictions=None, hyperparam=None) -> list:
        """ Acquisition function returning a randomly shuffled list for comparison """
        list_random = list(range(len(self.unvisited_param_configs)))
        shuffle(list_random)
        return list_random

    def get_diff_improvement(self, y_mu, y_std, fplus) -> torch.Tensor:
        """ compute probability of improvement by assuming normality on the difference in improvement """
        diff_improvement = (y_mu - fplus) / y_std    # y_std can be very small, causing diff_improvement to be very large
        diff_improvement = (diff_improvement - diff_improvement.mean()) / diff_improvement.std()    # force to N(0,1) with z-score
        if self.optimization_direction == 'max':
            diff_improvement = -diff_improvement
        return diff_improvement

    def contextual_variance(self, mean: torch.Tensor, std: torch.Tensor):
        """ Contextual improvement to decide explore / exploit, based on CI proposed by (Jasrasaria, 2018) """
        if not self.af_params['explorationfactor'] == 'CV':
            raise ValueError(f"Contextual Variance was called, but is not set as the exploration factor ({self.af_params['explorationfactor']})")
        if self.optimization_direction == 'max':
            raise NotImplementedError("Contextual Variance has not yet been implemented for maximisation")
        if self.current_optimum == self.inf_value:
            return 0.01
        if self.scaled_output:
            improvement_over_initial_sample = (abs(self.current_optimum) - self.initial_sample_mean) / self.initial_sample_std
            improvement_over_current_sample = (abs(self.current_optimum) - self.train_y.mean().item()) / std.mean().item()
            improvement_diff = improvement_over_current_sample - improvement_over_initial_sample
            # the closer the improvement over the current sample is to the improvement over the initial sample, the greater the exploration
            x = 1 - min(max(1 - improvement_diff, 0.2), 0.0)
            cv = np.log10(x) + 0.1    # at x=0.0, y=0.1; at x=0.2057, y=0.0.
            return cv
        else:
            raise NotImplementedError("Contextual Variance has not yet been implemented for non-scaled outputs")

    def af_probability_of_improvement_tensor(self, predictions: Tuple[torch.Tensor, torch.Tensor], hyperparam=None) -> torch.Tensor:
        """ Acquisition function Probability of Improvement (PoI) tensor-based """

        # prefetch required data
        y_mu, y_std = predictions
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        fplus = self.current_optimum - hyperparam

        diff_improvement = self.get_diff_improvement(y_mu, y_std, fplus)
        normal = torch.distributions.Normal(torch.zeros_like(diff_improvement), torch.ones_like(diff_improvement))
        cdf = normal.cdf(diff_improvement)

        # sanity check
        if torch.all(cdf == cdf[0]):
            raise ValueError("You need to scale the diff_improvement-values!")
        return cdf

    def af_expected_improvement_tensor(self, predictions: Tuple[torch.Tensor, torch.Tensor], hyperparam=None) -> torch.Tensor:
        """ Acquisition function Expected Improvement (EI) tensor-based """

        # prefetch required data
        y_mu, y_std = predictions
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        fplus = self.current_optimum - hyperparam

        diff_improvement = self.get_diff_improvement(y_mu, y_std, fplus)
        normal = torch.distributions.Normal(torch.zeros_like(diff_improvement), torch.ones_like(diff_improvement))
        cdf = normal.cdf(diff_improvement)
        pdf = torch.exp(normal.log_prob(diff_improvement))

        # sanity check
        if torch.all(cdf == cdf[0]) or torch.all(pdf == pdf[0]):
            raise ValueError("You need to scale the diff_improvement-values!")

        # compute expected improvement in bulk
        exp_improvement = (pdf + diff_improvement + y_std * cdf)
        # alternative exp_improvement = y_std * (pdf + diff_improvement * cdf)
        # alternative exp_improvement = -((fplus - y_mu) * cdf + y_std * pdf)
        return exp_improvement

    """                  """
    """ Helper functions """
    """                  """

    def get_hyperparam(self, name: str, default, supported_values=list()):
        """ Retrieve the value of a hyperparameter based on the name """
        value = self.tuning_options.strategy_options.get(name, default)
        if len(supported_values) > 0 and value not in supported_values:
            raise ValueError(f"'{name}' is set to {value}, but must be one of {supported_values}")
        return value

    def set_acquisition_function(self, acquisition_function: str):
        """ Set the acquisition function based on the name """
        if acquisition_function not in supported_methods:
            raise ValueError("Acquisition function must be one of {}, is {}".format(self.supported_methods, acquisition_function))

        if acquisition_function == 'poi':
            self.acquisition_function = self.af_probability_of_improvement_tensor
        elif acquisition_function == 'ei':
            self.acquisition_function = self.af_expected_improvement_tensor
        elif acquisition_function == 'random':
            self.acquisition_function = self.af_random

    def apply_scaling_to_inputs(self):
        """ Scale the inputs using min-max normalization (0-1) and remove constant parameters """
        param_configs_scaled = torch.zeros_like(self.param_configs)

        # first get the scaling factors of each parameter
        v_min_list = list()
        v_diff_list = list()
        unchanging_params_list = list()
        for param_values in self.tune_params.values():
            v_min = min(param_values)
            v_max = max(param_values)
            v_min_list.append(v_min)
            v_diff_list.append(v_max - v_min)
            unchanging_params_list.append(v_min == v_max)

        # then set each parameter value to the scaled value
        for param_index in range(len(self.param_configs[0])):
            v_min = v_min_list[param_index]
            v_diff = v_diff_list[param_index]
            param_configs_scaled[:,param_index] = torch.sub(self.param_configs[:,param_index], v_min).div(v_diff)

        # finally remove parameters that are constant by applying a mask
        unchanging_params_tensor = ~torch.tensor(unchanging_params_list, dtype=torch.bool)
        if torch.all(unchanging_params_tensor == False):
            raise ValueError(f"All of the parameter configurations ({self.size}) are the same: {self.param_configs[0]}, nothing to optimize")
        nonstatic_param_count = torch.count_nonzero(unchanging_params_tensor)
        self.param_configs_scaled = torch.zeros([len(param_configs_scaled), nonstatic_param_count], dtype=self.dtype)
        for param_config_index, param_config in enumerate(param_configs_scaled):
            self.param_configs_scaled[param_config_index] = param_config[unchanging_params_tensor]

    def transform_nonnumerical_params(self, parameter_space: list) -> Tuple[torch.Tensor, dict]:
        """ transform non-numerical or mixed-type parameters to numerical Tensor, also return new tune_params """
        parameter_space = deepcopy(parameter_space)
        number_of_params = len(parameter_space[0])

        # find out which parameters have nonnumerical or mixed types, and create a range of integers instead
        nonnumericals_exist = False
        nonnumerical_type = torch.zeros(number_of_params, dtype=torch.bool)
        nonnumerical_values = [ [] for _ in range(number_of_params) ]
        tune_params = deepcopy(self.tuning_options.tune_params)
        for param_index, (param_key, param_values) in enumerate(self.tuning_options.tune_params.items()):
            if not all(isinstance(v, (int, float, complex)) for v in param_values):
                nonnumericals_exist = True
                nonnumerical_type[param_index] = True
                nonnumerical_values[param_index] = param_values
                tune_params[param_key] = range(len(param_values))

        # overwrite the nonnumerical parameters with numerical parameters
        if nonnumericals_exist:
            self.tuning_options["snap"] = False     # snapping is only possible with numerical values
            for param_config_index, param_config in enumerate(parameter_space):
                parameter_space[param_config_index] = list(param_config)
                for param_index, param_value in enumerate(param_config):
                    if nonnumerical_type[param_index]:
                        # just use the index of the non-numerical value instead of the value
                        new_value = nonnumerical_values[param_index].index(param_value)
                        parameter_space[param_config_index][param_index] = new_value

        return torch.tensor(parameter_space, dtype=self.dtype).to(self.device), tune_params


    def visualize(self):
        """ Visualize the surrogate model and observations in a plot """
        from matplotlib import pyplot as plt
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.set_ylabel('Value')
            ax.set_xlabel('Parameter')

            param_configs = self.param_configs.to(self.out_device)

            # get true function
            objective_results = np.array([])
            for param_config in param_configs:
                result = self.objective_function(tuple(param_config.tolist()))
                if result == self.invalid_value:
                    result = np.nan
                objective_results = np.append(objective_results, result)
            if self.scaled_output:
                objective_results = (objective_results - objective_results.mean()) / objective_results.std()

            if len(param_configs[0]) == 1:
                ax.plot(np.linspace(param_configs[0], param_configs[-1], self.size), objective_results, 'r')
            else:
                ax.plot(range(self.size), objective_results, 'r')

            # take the parameter values for 1D, otherwise the indices
            if len(param_configs[0]) == 1:
                x_axis_param_configs = param_configs
                test_x_x_axis = self.test_x_unscaled.squeeze().to(self.out_device).numpy()
            else:
                x_axis_param_configs = torch.arange(self.size)
                test_x_x_axis = x_axis_param_configs[self.unvisited_configs].to(self.out_device)

            # Get upper and lower confidence bounds
            observed_pred = self.likelihood(self.model(self.test_x))
            lower, upper = observed_pred.confidence_region()
            lower, upper = lower.to(self.out_device), upper.to(self.out_device)

            # Plot initial sample as green stars
            initial_sample_x_axis = x_axis_param_configs[self.inital_sample_configs].to(self.out_device)
            initial_sample_y_axis = self.results[self.inital_sample_configs].to(self.out_device)
            ax.plot(initial_sample_x_axis.numpy(), initial_sample_y_axis.numpy(), 'g*')

            # Plot training data as black stars
            mask_training_data_no_initial_sample = ~self.inital_sample_configs == self.valid_configs
            training_x_axis = x_axis_param_configs[mask_training_data_no_initial_sample].to(self.out_device)
            training_y_axis = self.results[mask_training_data_no_initial_sample].to(self.out_device)
            ax.plot(training_x_axis.numpy(), training_y_axis.numpy(), 'k*')

            # Plot predictive means as blue line
            test_x_y_axis = observed_pred.mean.to(self.out_device)
            ax.plot(test_x_x_axis, test_x_y_axis.numpy(), 'b')

            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x_x_axis, lower.numpy(), upper.numpy(), alpha=0.5)

            # set the limits and legend
            # ax.set_ylim(min(objective_results), max(filter(lambda x: x != self.invalid_value, objective_results)))
            ax.legend(['Objective Function', 'Initial Sample', 'Observed Data', 'Mean', 'Confidence'])

            if self.animate:
                f.canvas.draw()
                plt.pause(0.1)

            plt.show()
