"""Lean implementation of Bayesian Optimization with GPyTorch."""
# python
import ast  # for casting strings to dict
import warnings
from copy import deepcopy
from math import ceil
from random import choice, randint, shuffle
from typing import Tuple

# external
import numpy as np
from numpy.random import default_rng

from kernel_tuner.runners.runner import Runner
from kernel_tuner.searchspace import Searchspace

# optional
try:
    import gpytorch
    import torch
    # import arviz as az
    bayes_opt_present = True

    from torch import Tensor

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
except ImportError:
    bayes_opt_present = False

    class Tensor():
        pass

    class ExactGPModel():
        def __init__(self, train_x, train_y, likelihood):
            raise ImportError("GPyTorch not imported")
        def forward(self, x):
            raise ImportError("GPyTorch not imported")


# set supported hyperparameter values
supported_precisions = ['float', 'double']
supported_initial_sample_methods = ['lhs', 'index', 'minmax','random']
supported_methods = ['ei', 'poi', 'random']
supported_cov_kernels = ['matern', 'matern_scalekernel']
supported_likelihoods = ['Gaussian', 'GaussianPrior', 'FixedNoise']
supported_optimizers = ['LBFGS', 'Adam', 'AdamW', 'Adagrad', 'ASGD']


# set complex hyperparameter defaults
def default_optimizer_learningrates(key):
    defaults = {
        'LBFGS': 1,
        'Adam': 0.001,
        'AdamW': 0.001,
        'ASGD': 0.01,
        'Adagrad': 0.01
    }
    return defaults[key]


def tune(searchspace: Searchspace, runner: Runner, tuning_options):
    """Find the best performing kernel configuration in the parameter space.

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
        raise ImportError(
            "Error: optional dependencies for Bayesian Optimization not installed, please install torch and gpytorch"
        )

    # set CUDA availability
    use_cuda = False
    cuda_available = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda:0" if cuda_available else "cpu")
    if cuda_available:
        print(f"CUDA is available, device: {torch.cuda.get_device_name(device)}")

    # retrieve options with defaults
    options = tuning_options.strategy_options
    optimization_direction = options.get("optimization_direction", 'min')
    num_initial_samples = int(options.get("popsize", 20))
    max_fevals = int(options.get("max_fevals", 220))

    # enabling scaling will unscale and snap inputs on evaluation, more efficient to scale all at once and keep unscaled values
    tuning_options["snap"] = False
    tuning_options["scaling"] = False

    # prune the search space using restrictions
    parameter_space = searchspace.list.copy()

    # limit max_fevals to max size of the parameter space
    max_fevals = min(len(parameter_space), max_fevals)
    if max_fevals < num_initial_samples:
        raise ValueError(
            f"Maximum number of function evaluations ({max_fevals}) can not be lower than or equal to the number of initial samples ({num_initial_samples}), you might as well brute-force."
        )

    # execute Bayesian Optimization
    BO = BayesianOptimization(parameter_space, tuning_options, runner, num_initial_samples, optimization_direction, device)
    all_results = BO.optimize(max_fevals)

    return all_results, runner.dev.get_environment()


class BayesianOptimization:

    def __init__(self, parameter_space: list, tuning_options, runner: Runner, num_initial_samples: int, optimization_direction: str,
                 device) -> None:
        self.animate = False    # TODO remove

        # set defaults
        self.num_initial_samples = num_initial_samples
        self.fevals = 0
        self.all_results = []
        self.unique_results = {}
        self.current_optimal_config = None

        # set Kernel Tuner data
        self.tuning_options = tuning_options
        self.runner = runner
        self.max_threads = runner.dev.max_threads

        # get precision options
        self.dtype = torch.float if self.get_hyperparam("precision", "float", supported_precisions) == "float" else torch.double
        self.min_std = self.get_hyperparam("minimum_std", 1e-6, type=float)

        # get tuning options
        self.initial_sample_method = self.get_hyperparam("initialsamplemethod", "lhs", supported_initial_sample_methods)
        self.initial_sample_random_offset_factor = self.get_hyperparam("initialsamplerandomoffsetfactor", 0.1, type=float)    # 0.1
        self.initial_training_iter = self.get_hyperparam("initialtrainingiter", 5, type=int)    # 5
        self.training_after_iter = self.get_hyperparam("trainingafteriter", 1, type=int)    # 1
        self.cov_kernel_name = self.get_hyperparam("covariancekernel", "matern_scalekernel", supported_cov_kernels)
        self.cov_kernel_lengthscale = self.get_hyperparam("covariancelengthscale", 1.5, type=float)
        self.likelihood_name = self.get_hyperparam("likelihood", "Gaussian", supported_likelihoods)
        self.optimizer_name = self.get_hyperparam("optimizer", "LBFGS", supported_optimizers)
        self.optimizer_learningrate = self.get_hyperparam("optimizer_learningrate", self.optimizer_name, type=float, cast=default_optimizer_learningrates)
        acquisition_function_name = self.get_hyperparam("method", "ei", supported_methods)
        af_params = self.get_hyperparam("methodparams", {}, type=dict, cast=ast.literal_eval)

        # set acquisition function options
        self.set_acquisition_function(acquisition_function_name)
        if 'explorationfactor' not in af_params:
            af_params['explorationfactor'] = 0.1    # 0.1
        self.af_params = af_params

        # set Tensors
        self.device: torch.device = device
        self.out_device = torch.device("cpu")
        self.size = len(parameter_space)
        self.index_counter = torch.arange(self.size)
        # the unvisited_configs and valid_configs are to be used as boolean masks on the other tensors, more efficient than adding to / removing from tensors
        self.unvisited_configs = torch.ones(self.size, dtype=torch.bool).to(device)
        self.valid_configs = torch.zeros(self.size, dtype=torch.bool).to(device)
        self.inital_sample_configs = torch.zeros(self.size, dtype=torch.bool).to(device)
        self.results = torch.zeros(self.size, dtype=self.dtype).to(device) * np.nan    # x (param configs) and y (results) must be the same type
        self.results_std = torch.ones(self.size, dtype=self.dtype).to(device)    # only a valid assumption if outputs are normalized

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
        self.hyperparams_means = {
            'loss': np.array([]),
            'lengthscale': np.array([]),
            'noise': np.array([]),
        }

        # initialize the model
        if not self.runner.simulation_mode:
            self.import_cached_evaluations()
        self.initialize_model()

    @property
    def train_x(self):
        """Get the valid parameter configurations."""
        return self.param_configs_scaled[self.valid_configs].to(self.device)

    @property
    def train_y(self):
        """Get the valid results."""
        outputs = self.results[self.valid_configs]
        if self.scaled_output:
            # z-score, remove mean and make unit variance to scale it to N(0,1)
            # alternatively, first min-max the outputs between -1 and +1 and apply a Fisher transformation (np.arctanh)
            outputs = (outputs - outputs.mean()) / outputs.std()
        return outputs

    @property
    def train_y_err(self):
        """Get the error on the valid results."""
        std = self.results_std[self.valid_configs]
        if self.scaled_output and std.std() > 0.0:
            std = (std - std.mean()) / std.std()    # use z-score to get normalized variability
        return std

    @property
    def test_x(self):
        """Get the not yet visited parameter configurations."""
        return self.param_configs_scaled[self.unvisited_configs].to(self.device)

    @property
    def test_x_unscaled(self):
        """Get the unscaled, not yet visited parameter configurations."""
        return self.param_configs[self.unvisited_configs]

    @property
    def test_y_err(self):
        """Get the expected error on the test set."""
        train_y_err = self.train_y_err
        return torch.full((self.size - len(train_y_err), ), torch.mean(train_y_err))

    @property
    def invalid_x(self):
        """Get the invalid parameter configurations by checking which visited configs are not valid (equivalent to checking which unvisited configs are valid)."""
        invalid_mask = (self.unvisited_configs == self.valid_configs)
        return self.param_configs[invalid_mask]

    def true_param_config_index(self, target_index: int) -> int:
        """The index required to get the true config param index when dealing with test_x."""
        # get the index of the #index-th True (for example the 9th+1 True could be index 13 because there are 4 Falses in between)
        masked_counter = self.index_counter[self.unvisited_configs]
        return masked_counter[target_index]

    def true_param_config_indices(self, target_indices: Tensor) -> Tensor:
        """Same as true_param_config_index, but for an array of targets instead."""
        masked_counter = self.index_counter[self.unvisited_configs]
        return masked_counter.index_select(0, target_indices)

    def initialize_model(self, take_initial_sample=True, train_hyperparams=True):
        """Initialize the surrogate model."""
        # self.initial_sample_std = self.min_std
        if take_initial_sample:
            self.initial_sample()

        # create the model
        if self.likelihood_name == 'Gaussian':
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        elif self.likelihood_name == 'GaussianPrior':
            raise NotImplementedError("Gaussian Prior likelihood has not been implemented yet")
        elif self.likelihood_name == 'FixedNoise':
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=self.train_y_err.clamp(min=self.min_std), learn_additional_noise=True)
        self.likelihood = self.likelihood.to(self.device)
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood, self.cov_kernel_name, self.cov_kernel_lengthscale)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        # set the optimizer
        # LBFGS is probably better as Adam is first-order
        if self.optimizer_name == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(model_parameters, lr=self.optimizer_learningrate)
        elif self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(model_parameters, lr=self.optimizer_learningrate)
        elif self.optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(model_parameters, lr=self.optimizer_learningrate)
        elif self.optimizer_name == 'ASGD':
            self.optimizer = torch.optim.ASGD(model_parameters, lr=self.optimizer_learningrate)
        elif self.optimizer_name == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(model_parameters, lr=self.optimizer_learningrate)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.device)
        if train_hyperparams:
            self.train_hyperparams(self.initial_training_iter)
        else:
            self.train_hyperparams(0)

    def import_cached_evaluations(self):
        """Import the previously evaluated configurations into this run."""
        # make strings of all the parameter configurations in the search space
        param_config_strings = list()
        for param_config in self.true_param_configs:
            param_config_strings.append(",".join([str(v) for v in param_config]))

        # load the results from the cache into the run
        cache = self.tuning_options.cache
        if len(cache.keys()) > 0:
            print("Previous cachefile found while not in simulation mode, importing previous evaluations.")
        for param_config_string, result in cache.items():
            # get the index of the string in the search space
            param_config_index = param_config_strings.index(param_config_string)
            time = self.evaluate_config(param_config_index)
            assert time == result['time']
        print(f"Imported {len(self.all_results)} previously evaluated configurations.")

    def initial_sample(self):
        """Take an initial sample of the parameter space."""
        list_param_config_indices = list(self.index_counter[~self.unvisited_configs])

        # generate a random offset from a normal distribution to add to the sample indices
        rng = default_rng()
        if self.initial_sample_random_offset_factor > 0.5:
            raise ValueError("Random offset factor should not be greater than 0.5 to avoid overlapping index offsets")
        random_offset_size = (self.size / self.num_initial_samples) * self.initial_sample_random_offset_factor
        random_offsets = np.round(rng.standard_normal(self.num_initial_samples) * random_offset_size)

        # first apply the initial sampling method
        if self.initial_sample_method == 'lhs' and self.num_initial_samples - self.fevals > 1:
            indices = self.get_lhs_samples(random_offsets)
            for param_config_index in indices.tolist():
                if param_config_index in list_param_config_indices:
                    continue
                list_param_config_indices.append(param_config_index)
                self.evaluate_config(param_config_index)
        elif self.initial_sample_method == 'random':
            while self.fevals < self.num_initial_samples:
                param_config_index = randint(0, self.size - 1)
                if param_config_index in list_param_config_indices:
                    continue
                list_param_config_indices.append(param_config_index)
                self.evaluate_config(param_config_index)
        elif self.initial_sample_method == 'minmax':
            list_param_config_indices += self.take_min_max_initial_samples(list_param_config_indices)

        # then take index-spaced samples until all samples are valid
        while self.fevals < self.num_initial_samples:
            least_evaluated_region_index = self.get_middle_index_of_least_evaluated_region()
            param_config_index = min(max(int(least_evaluated_region_index + random_offsets[self.fevals].item()), 0), self.size - 1)
            if param_config_index in list_param_config_indices:
                warnings.warn(
                    f"An already evaluated configuration ({param_config_index}) was selected for index-spaced sampling. " +
                    "If this happens regularly, reduce the initial sample random offset factor.", AlreadyEvaluatedConflict)
                param_config_index = least_evaluated_region_index
            list_param_config_indices.append(param_config_index)
            self.evaluate_config(param_config_index)

        # set the current optimum, initial sample mean and initial sample std
        self.current_optimum = self.opt(self.train_y).item()
        self.initial_sample_mean = self.train_y.mean().item()
        self.initial_sample_std = self.train_y.std().item()
        # self.initial_sample_std = self.min_std    # temporary until the predictive posterior has been taken

        # save a boolean mask of the initial samples
        self.inital_sample_configs = self.valid_configs.detach().clone()

    def get_lhs_samples(self, random_offsets: np.ndarray) -> Tensor:
        """Get a centered Latin Hypercube Sample with a random offset."""
        n_samples = self.num_initial_samples - self.fevals

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
        param_configs = param_configs.unique(dim=0)    # remove duplicates
        n_samples_unique = len(param_configs)

        # get the indices of the parameter configurations
        num_params = len(self.param_configs[0])
        minimum_required_num_matching_params = round(num_params *
                                                     0.75)    # set the number of parameter matches allowed to be dropped before the search is stopped
        param_configs_indices = torch.full((n_samples_unique, ), -1, dtype=torch.int)
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
            param_configs_indices[selected_index] = min(max(int(index + random_offsets[selected_index].item()), 0), self.size - 1)

        # filter -1 indices and duplicates that occurred because of the random offset
        param_configs_indices = param_configs_indices[param_configs_indices >= 0]
        param_configs_indices = param_configs_indices.unique().type(torch.int)
        if len(param_configs_indices) < n_samples / 2:
            warnings.warn(
                str(f"{n_samples - len(param_configs_indices)} out of the {n_samples} LHS samples were duplicates or -1." +
                    f"This might be because you have few initial samples ({n_samples}) relative to the number of parameters ({num_params})." +
                    "Perhaps try something other than LHS."))
        return param_configs_indices

    def take_min_max_initial_samples(self, list_param_config_indices: list, samples_per_parameter=1) -> list:
        """Take the minimum parameters and the maximum for each parameter to establish the effect of individual parameters."""
        # number of samples required is at least (samples_per_parameter) * (number of parameters) + 1

        # first get the individual parameter values and sort them
        params_values = list(self.tune_params.values())
        for param_values in params_values:
            param_values.sort()

        number_of_params = len(params_values)
        if self.num_initial_samples - self.fevals < samples_per_parameter * number_of_params + 1:
            raise ValueError(f"There are not enough initial samples available ({self.num_initial_samples - self.fevals}) to do minmax initial sampling. At least {samples_per_parameter * number_of_params + 1} samples are required.")

        # then take the minimum parameter configuration using BFS, this is used as the base
        # instead of BFS, you could also search for the minimal sum of indices
        minimum_index = None
        param_level = 0
        param_moving_index = -1
        while minimum_index is None and self.num_initial_samples - self.fevals:
            # create the minimum base configuration and find it in the search space
            selected_param_config = torch.tensor(tuple(param_values[param_level+1] if param_index == param_moving_index else param_values[min(param_level, len(param_values)-1)] for param_index, param_values in enumerate(params_values)), dtype=self.dtype).to(self.device)
            matching_params = torch.count_nonzero(self.param_configs == selected_param_config, -1)
            match_mask = (matching_params == number_of_params)
            found_num_matching_param_configs = match_mask.count_nonzero()
            temp_index = self.index_counter[match_mask]
            # check if the configuration exists and is succesfully evaluated
            if found_num_matching_param_configs == 1 and (temp_index.item() in list_param_config_indices or self.evaluate_config(temp_index.item()) < self.invalid_value):
                minimum_index = temp_index.item()
                minimum_config = self.param_configs[minimum_index]
                if minimum_index not in list_param_config_indices:
                    list_param_config_indices.append(minimum_index)
            # if it doesn't exist and evaluate, do a breadth-first search for the minimum configuration
            else:
                proceed = False
                while not proceed:
                    # first look at the current level
                    if param_moving_index < len(params_values) - 1:
                        param_moving_index += 1
                        # if the param_level + 1 exceeds the number of parameters, try the next parameter
                        if len(params_values[param_moving_index]) <= param_level + 1:
                            param_moving_index += 1
                        else:
                            proceed = True
                    # if nothing is found, proceed to the next level
                    else:
                        param_level += 1
                        param_moving_index = -1
                        proceed = True
        if minimum_index is None:
            raise ValueError(f"Could not evaluate the minimum base configuration in {self.num_initial_samples} samples.")

        # next take the maximum for each individual parameter using DFS
        for param_index, param_values in enumerate(params_values):
            if len(param_values) <= 1:
                continue
            maximum_index = None
            param_moving_level = len(param_values) - 1
            while maximum_index is None and self.num_initial_samples - self.fevals > 0:
                # take the minimum configuration as base
                selected_param_config = minimum_config.clone()
                # change only the currently selected parameter and look up the configuration in the search space
                selected_param_config[param_index] = param_values[param_moving_level]
                matching_params = torch.count_nonzero(self.param_configs == selected_param_config, -1)
                match_mask = (matching_params == number_of_params)
                found_num_matching_param_configs = match_mask.count_nonzero()
                temp_index = self.index_counter[match_mask]
                if found_num_matching_param_configs == 1 and (temp_index.item() in list_param_config_indices or self.evaluate_config(temp_index.item()) < self.invalid_value):
                    maximum_index = temp_index.item()
                    if maximum_index not in list_param_config_indices:
                        list_param_config_indices.append(maximum_index)
                # if it doesn't exist and evaluate, move one parameter value down
                else:
                    param_moving_level -= 1
                    if param_moving_level < 0:
                        raise ValueError(f"No instance of parameter {param_index} is present in the search space and succesfully evaluated")
            if maximum_index is None:
                raise ValueError(f"Could not evaluate the maximum configuration for {param_index+1} out of {len(params_values)} within {self.num_initial_samples} samples.")

        return list_param_config_indices

    def get_middle_index_of_least_evaluated_region(self) -> int:
        """Get the middle index of the region of parameter configurations that is the least visited."""
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

    def train_hyperparams(self, training_iter: int):
        """Optimize the surrogate model hyperparameters iteratively."""
        self.model.train()
        self.likelihood.train()

        def closure():
            self.optimizer.zero_grad()
            output = self.model(self.train_x)    # get model output
            try:
                loss = -self.mll(output, self.train_y)    # calculate loss and backprop gradients
                loss.backward()
                # large sudden increase in loss signals numerical instability
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    no_nan_losses = self.hyperparams_means['loss'][~np.isnan(self.hyperparams_means['loss'])]
                    if len(no_nan_losses) > 1 and loss.item() > np.mean(no_nan_losses) * 2:
                        warnings.warn("Avoiding loss surge, aborting training", AvoidedLossSurgeWarning)
                        return np.nan
                return loss
            except gpytorch.utils.errors.NotPSDError:
                warnings.warn("Matrix not positive definite during training", NotPSDTrainingWarning)
                return np.nan
            except RuntimeError as e:
                warnings.warn(str(e), RuntimeWarning)

        loss = None
        for _ in range(training_iter):
            try:
                _loss = self.optimizer.step(closure)
                if _loss is np.nan:
                    break
                loss = _loss
            except gpytorch.utils.errors.NanError:
                warnings.warn("PSD_safe_Cholesky failed due to too many NaN", NaNTrainingWarning)
                break
            except TypeError as e:
                warnings.warn(str(e), RuntimeWarning)
                break

        # set the hyperparams to the new values
        try:
            lengthscale = float(self.model.covar_module.lengthscale.item())
        except AttributeError:
            lengthscale = float(self.model.covar_module.base_kernel.lengthscale.item())
        loss = float(loss.item()) if loss is not None else np.nan
        noise = float(self.model.likelihood.noise.mean().detach())
        self.hyperparams = {
            'loss': loss,
            'lengthscale': lengthscale,
            'noise': noise,
        }
        self.hyperparams_means['loss'] = np.append(self.hyperparams_means['loss'], loss)
        self.hyperparams_means['lengthscale'] = np.append(self.hyperparams_means['lengthscale'], lengthscale)
        self.hyperparams_means['noise'] = np.append(self.hyperparams_means['noise'], noise)

        # get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

    def optimize(self, max_fevals: int) -> Tuple[tuple, float]:    #NOSONAR
        """Optimize the objective."""
        predictions_tuple = None
        short_param_config_index = None
        last_invalid = False
        report_multiple_minima = ceil(round(self.size / 10))    # if more than 10% of the space is minima, print a warning
        use_contextual_variance = self.af_params['explorationfactor'] == 'CV'
        while self.fevals < max_fevals:
            if last_invalid:
                # TODO no need to get the predictions again as the predictions are unchanged, just set the invalid param config mean to the worst non-NAN value and the std to 0
                # predictions_tuple[0][short_param_config_index] = torch.nanmean(predictions_tuple[0])
                # predictions_tuple[1][short_param_config_index] = 0
                predictions_tuple = self.remove_from_predict_list(predictions_tuple, short_param_config_index)
            else:
                predictions_tuple = self.predict_list()
                # if self.initial_sample_std <= self.min_std:
                # self.initial_sample_std = min(max(predictions_tuple[1].mean().item(), self.min_std), 10.0)
            # if there are NaN or all of the predicted std are the same, take from the least evaluated region
            mean_has_NaN = bool(torch.any(torch.isnan(predictions_tuple[0])).item())
            std_has_NaN = bool(torch.any(torch.isnan(predictions_tuple[1])).item())
            if mean_has_NaN or std_has_NaN or torch.all(predictions_tuple[1] == predictions_tuple[1][0]):
                least_evaluated_region_index = self.get_middle_index_of_least_evaluated_region()
                param_config_index = least_evaluated_region_index
                short_param_config_index = -1
                if mean_has_NaN:
                    warning_reason = "there were NaN in the predicted mean"
                elif std_has_NaN:
                    warning_reason = "there were NaN in the predicted std"
                else:
                    warning_reason = "all STDs were the same"
                warnings.warn(
                    f"After {self.fevals}/{max_fevals} fevals, {warning_reason}, picking one from the least evaluated region and resetting the surrogate model",
                    ResetModelWarning)
                self.initialize_model(take_initial_sample=False, train_hyperparams=True)
            else:
                # otherwise, optimize the acquisition function to find the next candidate
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
                        warnings.warn(
                            f"After {self.fevals}/{max_fevals} fevals, there were multiple minima in the acquisition values ({len(indices_where_min)}), picking one based on the least evaluated region",
                            MultipleMinimaWarning)

            # evaluate and register the result
            result = self.evaluate_config(param_config_index)
            if result == self.invalid_value and short_param_config_index > -1:
                # can't use last_invalid if short_param_config_index is not set
                last_invalid = True
            else:
                last_invalid = False
                self.model.set_train_data(self.train_x, self.train_y, strict=False)
                # do not train if there are multiple minima, because it introduces numerical instability or insolvability
                if self.training_after_iter > 0 and (self.fevals % self.training_after_iter == 0):
                    self.train_hyperparams(training_iter=1)    # TODO experiment with other training iter
                # set the current optimum
                self.current_optimum = self.opt(self.train_y).item()
            # print(f"Valid: {len(self.train_x)}, unvisited: {len(self.test_x)}, invalid: {len(self.invalid_x)}, last invalid: {last_invalid}")
            if self.animate:
                self.visualize()

        return self.all_results

    def objective_function(self, param_config: tuple) -> float:
        return self.runner.run([param_config], self.tuning_options)

    def evaluate_config(self, param_config_index: int) -> float:
        """Evaluates a parameter configuration, returns the time."""
        param_config = self.true_param_configs[param_config_index]
        time = self.objective_function(param_config)
        self.register_result(time, param_config_index)
        self.update_unique_results()
        self.fevals = len(self.unique_results)
        return time

    def register_result(self, result: float, param_config_index: int):
        """Registers the result to the Tensors and adds the hyperparameters to the results dict."""
        # set the unvisited Tensors
        if self.unvisited_configs[param_config_index] is False:
            raise ValueError(f"The param config index {param_config_index} was already set to False!")
        self.unvisited_configs[param_config_index] = False

        # set the results Tensors
        last_result = self.all_results[-1]
        if result != self.invalid_value:
            self.valid_configs[param_config_index] = True
            self.results[param_config_index] = result
            # assert last_result['time'] == result TODO remove
            self.results_std[param_config_index] = max(np.std(last_result['times']), self.min_std)

        # add the current model parameters to the last entry of the results dict
        if len(self.all_results) < 1:
            return
        for key, value in self.hyperparams.items():
            last_result["hyperparam_" + key] = value
        self.all_results[-1] = last_result
        # TODO check if it is possible to write the results with hyperparameters to the cache if not in simulation mode, maybe with observer?

    def update_unique_results(self):
        """Updates the unique results dictionary."""
        record = self.all_results[-1]
        # make a unique string by taking every value in a result, if it already exists, it is overwritten
        self.unique_results.update({",".join([str(v) for k, v in record.items() if k in self.tuning_options.tune_params]): record["time"]})

    def predict_list(self) -> Tuple[Tensor, Tensor]:
        """Returns the means and standard deviations predicted by the surrogate model for the unvisited parameter configurations."""
        with torch.no_grad(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.fast_pred_var():
            try:
                observed_pred = self.likelihood(self.model(self.test_x))
                mu = observed_pred.mean
                std = observed_pred.variance.clamp(min=self.min_std)    # TODO .sqrt() or not? looks like without is better
                return mu, std
            except gpytorch.utils.errors.NanError:
                warnings.warn("NaN error during predictions", NaNPredictionWarning)
                return torch.ones_like(self.test_x), torch.zeros_like(self.test_x)
            except gpytorch.utils.errors.NotPSDError:
                warnings.warn("NotPSD error during predictions", NotPSDPredictionWarning)
                return torch.ones_like(self.test_x), torch.zeros_like(self.test_x)
            except RuntimeError as e:
                warnings.warn(str(e), RuntimeWarning)
                return torch.ones_like(self.test_x), torch.zeros_like(self.test_x)

    def get_diff_improvement(self, y_mu, y_std, fplus) -> Tensor:
        """Compute probability of improvement by assuming normality on the difference in improvement."""
        diff_improvement = (y_mu - fplus) / y_std    # y_std can be very small, causing diff_improvement to be very large
        diff_improvement = (diff_improvement - diff_improvement.mean()) / max(diff_improvement.std(), self.min_std)    # force to N(0,1) with z-score
        if self.optimization_direction == 'max':
            diff_improvement = -diff_improvement
        return diff_improvement

    def contextual_variance(self, mean: Tensor, std: Tensor):
        """Contextual improvement to decide explore / exploit, based on CI proposed by (Jasrasaria, 2018)."""
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
            # x = 1 - max(max(1 - improvement_diff, 0.2), 0.0)
            x = 1 - max(min(improvement_diff, 1) * 0.2, 0.0)
            # the smaller the difference between the initial sample error and current sample error, the greater the exploration
            # x = 1 - min(max(self.initial_sample_std - std.mean().item(), 1.0), 0.8)
            # print(self.initial_sample_std, std.mean().item())
            cv = np.log10(x) + 0.1    # at x=0.0, y=0.1; at x=0.2, y=0.003; at x=0.2057, y=0.0.
            return cv
        else:
            raise NotImplementedError("Contextual Variance has not yet been implemented for non-scaled outputs")

    def af_random(self, predictions=None, hyperparam=None) -> list:
        """Acquisition function returning a randomly shuffled list for comparison."""
        list_random = list(range(len(self.unvisited_param_configs)))
        shuffle(list_random)
        return list_random

    def af_probability_of_improvement_tensor(self, predictions: Tuple[Tensor, Tensor], hyperparam=None) -> Tensor:
        """Acquisition function Probability of Improvement (PoI) tensor-based."""
        # prefetch required data
        y_mu, y_std = predictions
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        fplus = self.current_optimum - hyperparam

        diff_improvement = self.get_diff_improvement(y_mu, y_std, fplus)
        normal = torch.distributions.Normal(torch.zeros_like(diff_improvement), torch.ones_like(diff_improvement))
        cdf = normal.cdf(diff_improvement)

        # # sanity check
        # if torch.all(cdf == cdf[0]):
        #     raise FloatingPointError("You need to scale the diff_improvement-values!")
        return cdf

    def af_expected_improvement_tensor(self, predictions: Tuple[Tensor, Tensor], hyperparam=None) -> Tensor:
        """Acquisition function Expected Improvement (EI) tensor-based."""
        # prefetch required data
        y_mu, y_std = predictions
        if hyperparam is None:
            hyperparam = self.af_params['explorationfactor']
        fplus = self.current_optimum - hyperparam

        diff_improvement = self.get_diff_improvement(y_mu, y_std, fplus)
        normal = torch.distributions.Normal(torch.zeros_like(diff_improvement), torch.ones_like(diff_improvement))
        cdf = normal.cdf(diff_improvement)
        pdf = torch.exp(normal.log_prob(diff_improvement))

        # # sanity check
        # if torch.all(cdf == cdf[0]) and torch.all(pdf == pdf[0]):
        #     raise FloatingPointError("You need to scale the diff_improvement-values!")

        # compute expected improvement in bulk
        exp_improvement = (pdf + diff_improvement + y_std * cdf)
        # alternative exp_improvement = y_std * (pdf + diff_improvement * cdf)
        # alternative exp_improvement = -((fplus - y_mu) * cdf + y_std * pdf)
        return exp_improvement

    """                  """
    """ Helper functions """
    """                  """

    def apply_scaling_to_inputs(self):
        """Scale the inputs using min-max normalization (0-1) and remove constant parameters."""
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
            param_configs_scaled[:, param_index] = torch.sub(self.param_configs[:, param_index], v_min).div(v_diff)

        # finally remove parameters that are constant by applying a mask
        unchanging_params_tensor = ~torch.tensor(unchanging_params_list, dtype=torch.bool)
        # if torch.all(unchanging_params_tensor == False):
        # raise ValueError(f"All of the parameter configurations ({self.size}) are the same: {self.param_configs[0]}, nothing to optimize")
        nonstatic_param_count = torch.count_nonzero(unchanging_params_tensor)
        self.param_configs_scaled = torch.zeros([len(param_configs_scaled), nonstatic_param_count], dtype=self.dtype)
        for param_config_index, param_config in enumerate(param_configs_scaled):
            self.param_configs_scaled[param_config_index] = param_config[unchanging_params_tensor]
        self.nonstatic_params = unchanging_params_tensor

    def find_nearest(self, value, array: Tensor):
        """Find the value nearest to the given value in the array."""
        index = (torch.abs(array - value)).argmin()
        return array[index]

    def get_hyperparam(self, name: str, default, supported_values=list(), type=None, cast=None):
        """Retrieve the value of a hyperparameter based on the name - beware that cast can be a reference to any function."""
        value = self.tuning_options.strategy_options.get(name, default)

        # check with predifined value list
        if len(supported_values) > 0 and value not in supported_values:
            raise ValueError(f"'{name}' is set to {value}, but must be one of {supported_values}")
        # cast to type if provided
        if type and not isinstance(value, type):
            if cast:
                value = cast(value)
            else:
                value = type(value)

        # exceptions with more complex types
        if value == 'methodparams' and 'explorationfactor' in value and value['explorationfactor'] != 'CV':
            value = float(value)
        return value

    def remove_from_predict_list(self, p: Tuple[Tensor, Tensor], i: int) -> Tuple[Tensor, Tensor]:
        """Remove an index from a tuple of predictions."""
        return torch.cat([p[0][:i], p[0][i + 1:]]), torch.cat([p[1][:i], p[1][i + 1:]])

    def set_acquisition_function(self, acquisition_function: str):
        """Set the acquisition function based on the name."""
        if acquisition_function not in supported_methods:
            raise ValueError("Acquisition function must be one of {}, is {}".format(self.supported_methods, acquisition_function))

        if acquisition_function == 'poi':
            self.acquisition_function = self.af_probability_of_improvement_tensor
        elif acquisition_function == 'ei':
            self.acquisition_function = self.af_expected_improvement_tensor
        elif acquisition_function == 'random':
            self.acquisition_function = self.af_random

    def transform_nonnumerical_params(self, parameter_space: list) -> Tuple[Tensor, dict]:
        """Transform non-numerical or mixed-type parameters to numerical Tensor, also return new tune_params."""
        parameter_space = deepcopy(parameter_space)
        number_of_params = len(parameter_space[0])

        # find out which parameters have nonnumerical or mixed types, and create a range of integers instead
        nonnumericals_exist = False
        nonnumerical_type = torch.zeros(number_of_params, dtype=torch.bool)
        nonnumerical_values = [[] for _ in range(number_of_params)]
        tune_params = deepcopy(self.tuning_options.tune_params)
        for param_index, (param_key, param_values) in enumerate(self.tuning_options.tune_params.items()):
            if not all(isinstance(v, (int, float, complex)) for v in param_values):
                nonnumericals_exist = True
                nonnumerical_type[param_index] = True
                nonnumerical_values[param_index] = param_values
                tune_params[param_key] = range(len(param_values))

        # overwrite the nonnumerical parameters with numerical parameters
        if nonnumericals_exist:
            self.tuning_options["snap"] = False    # snapping is only possible with numerical values
            for param_config_index, param_config in enumerate(parameter_space):
                parameter_space[param_config_index] = list(param_config)
                for param_index, param_value in enumerate(param_config):
                    if nonnumerical_type[param_index]:
                        # just use the index of the non-numerical value instead of the value
                        new_value = nonnumerical_values[param_index].index(param_value)
                        parameter_space[param_config_index][param_index] = new_value

        return torch.tensor(parameter_space, dtype=self.dtype).to(self.device), tune_params

    def visualize(self):
        """Visualize the surrogate model and observations in a plot."""
        if self.fevals < 220:
            return None
        from matplotlib import pyplot as plt
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Initialize plot
            f = plt.figure(constrained_layout=True, figsize=(10, 8))
            subfigures = f.subfigures(2, 1)
            ax = subfigures[0].subplots(1, 1)
            axes2 = subfigures[1].subplots(1, 3)
            ax.set_ylabel('Value')
            ax.set_xlabel('Parameter')

            param_configs = self.true_param_configs

            # get true function
            objective_results = np.array([])
            for param_config in param_configs:
                result = self.objective_function(tuple(param_config))
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

            # draw the hyperparameter plots
            # loss
            axes2[0].plot(self.hyperparams_means['loss'])
            axes2[0].set_ylabel('Loss')
            axes2[0].set_xlabel('Number of evaluations')
            # lengthscale
            axes2[1].plot(self.hyperparams_means['lengthscale'])
            axes2[1].set_ylabel('Lengthscale')
            axes2[1].set_xlabel('Number of evaluations')
            # noise
            axes2[2].plot(self.hyperparams_means['noise'])
            axes2[2].set_ylabel('Noise')
            axes2[2].set_xlabel('Number of evaluations')

            if self.animate:
                # f.canvas.draw()
                plt.savefig('animation_last_graph')
                # plt.pause(0.1)

            # plt.show()


class CustomWarning(Warning):

    def __init__(self, message: str, category: str) -> None:
        # super().__init__()
        self.message = message
        self.category = category

    def __str__(self):
        return repr(self.message)

    def category(self):
        return self.category.__name__


class AvoidedLossSurgeWarning(CustomWarning):

    def __init__(self, message: str) -> None:
        super().__init__(message, "AvoidedLossSurgeWarning")


class NotPSDTrainingWarning(CustomWarning):

    def __init__(self, message: str) -> None:
        super().__init__(message, "NotPSDTrainingWarning")


class NaNTrainingWarning(CustomWarning):

    def __init__(self, message: str) -> None:
        super().__init__(message, "NaNTrainingWarning")


class NaNPredictionWarning(CustomWarning):

    def __init__(self, message: str) -> None:
        super().__init__(message, "NaNPredictionWarning")


class NotPSDPredictionWarning(CustomWarning):

    def __init__(self, message: str) -> None:
        super().__init__(message, "NotPSDPredictionWarning")


class ResetModelWarning(CustomWarning):

    def __init__(self, message: str) -> None:
        super().__init__(message, "ResetModelWarning")


class MultipleMinimaWarning(CustomWarning):

    def __init__(self, message: str) -> None:
        super().__init__(message, "MultipleMinimaWarning")


class AlreadyEvaluatedConflict(CustomWarning):

    def __init__(self, message: str) -> None:
        super().__init__(message, "AlreadyEvaluatedConflict")
