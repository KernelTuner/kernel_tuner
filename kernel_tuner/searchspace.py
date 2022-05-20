from random import choice, shuffle
from typing import Tuple, List

from constraint import Problem, Constraint, FunctionConstraint
import numpy as np

from kernel_tuner.util import default_block_size_names
from kernel_tuner.util import check_restrictions as check_instance_restrictions
from kernel_tuner.util import MaxProdConstraint

supported_neighbor_methods = ['strictly-adjacent', 'adjacent', 'Hamming']

class Searchspace():
    """ Class that offers the search space to strategies """

    def __init__(self, tuning_options: dict, max_threads: int, build_neighbors_index=False, neighbor_method=None, sort=False, sort_last_param_first=False) -> None:
        """ Build a searchspace using the variables and constraints.
            Optionally build the neighbors index - only faster if you repeatedly look up neighbors. Methods:
                strictly-adjacent: differs +1 or -1 parameter index value for each parameter
                adjacent: picks closest parameter value in both directions for each parameter
                Hamming: any parameter config with 1 different parameter value is a neighbor
            Optionally sort the searchspace by the order in which the parameter values were specified. By default, sort goes from first to last parameter, to reverse this use sort_last_param_first.
        """
        self.tuning_options = tuning_options
        self.restrictions = tuning_options.restrictions
        self.tune_params = tuning_options.tune_params
        self.max_threads = max_threads
        self.param_names = list(self.tune_params.keys())
        self.params_values = tuple(tuple(param_vals) for param_vals in self.tune_params.values())
        self.params_values_indices = None
        self.build_neighbors_index = build_neighbors_index
        self.__neighbor_cache = dict()
        self.neighbor_method = neighbor_method
        if (neighbor_method is not None or build_neighbors_index) and neighbor_method not in supported_neighbor_methods:
            raise ValueError(f"Neighbor method is {neighbor_method}, must be one of {supported_neighbor_methods}")

        self.list, self.__numpy, self.__dict, self.size = self.__build_searchspace(sort, sort_last_param_first)
        self.num_params = len(self.list[0])
        self.indices = np.arange(self.size)
        if neighbor_method is not None and neighbor_method != 'Hamming':
            self.__prepare_neighbors_index()
        if build_neighbors_index:
            self.neighbors_index = self.__build_neighbors_index(neighbor_method)

    def __build_searchspace(self, sort: bool, sort_last_param_first: bool) -> Tuple[List[tuple], np.ndarray, dict, int]:
        """ compute valid configurations in a search space based on restrictions and max_threads, returns the searchspace, a dict of the searchspace for fast lookups and the size """

        # instantiate the parameter space with all the variables
        parameter_space = Problem()
        for param_name, param_values in self.tune_params.items():
            parameter_space.addVariable(param_name, param_values)

        # add the user-specified restrictions as constraints on the parameter space
        parameter_space = self.__add_restrictions(parameter_space)

        # add the default blocksize threads restrictions last, because it is unlikely to reduce the parameter space by much
        block_size_names = self.tuning_options.get("block_size_names", default_block_size_names)
        block_size_names = list(block_size_name for block_size_name in block_size_names if block_size_name in self.param_names)
        if len(block_size_names) > 0:
            parameter_space.addConstraint(MaxProdConstraint(self.max_threads), block_size_names)

        # construct the parameter space with the constraints applied
        parameter_space = parameter_space.getSolutions()

        # form the parameter tuples in the order specified by tune_params.keys()
        parameter_space_list = list((tuple(params[param_name] for param_name in self.param_names)) for params in parameter_space)

        # in order to have the tuples as tuples in numpy, the types are set with a string, but this will make the type np.void
        # type_string = ",".join(list(type(param).__name__ for param in parameter_space_list[0]))
        parameter_space_numpy = np.array(parameter_space_list)

        # sort the parameter space on the order of parameters and their values as specified
        if sort is True:
            params_values_indices = list(self.get_param_indices(param_config) for param_config in parameter_space_list)
            params_values_indices_dict = dict(zip(params_values_indices, list(range(len(params_values_indices)))))

            # Python's built-in sort will sort starting in front, so if we want to vary the first parameter the tuple needs to be reversed
            params_values_indices.sort(key=lambda t: tuple(reversed(t))) if sort_last_param_first else params_values_indices.sort()

            # find the index of the parameter configuration for each parameter value index, using a dict to do it in constant time
            new_order = [params_values_indices_dict.get(param_values_indices) for param_values_indices in params_values_indices]
            # apply the new order
            parameter_space_list = [parameter_space_list[i] for i in new_order]

        # create a dictionary with the hashed parameter configurations as keys and indices as values for fast lookups
        parameter_space_dict = dict(zip(parameter_space_list, list(range(parameter_space_numpy.size))))

        # check for duplicates
        size_list = len(parameter_space_list)
        size_dict = len(parameter_space_dict.keys())
        if size_list != size_dict:
            raise ValueError(f"{size_list - size_dict} duplicate parameter configurations in the searchspace, this should not happen")

        return parameter_space_list, parameter_space_numpy, parameter_space_dict, size_list

    def __add_restrictions(self, parameter_space: Problem) -> Problem:
        """ add the user-specified restrictions as constraints on the parameter space """
        if isinstance(self.restrictions, list):
            for restriction in self.restrictions:
                if callable(restriction) and not isinstance(restriction, Constraint):
                    restriction = FunctionConstraint(restriction)
                if isinstance(restriction, FunctionConstraint):
                    parameter_space.addConstraint(restriction, self.param_names)
                elif isinstance(restriction, Constraint):
                    parameter_space.addConstraint(restriction)
                else:
                    raise ValueError(f"Unrecognized restriction {restriction}")

        # if the restrictions are the old monolithic function, apply them directly (only for backwards compatibility, likely slower than well-specified constraints!)
        elif callable(self.restrictions):
            restrictions_wrapper = lambda *args: check_instance_restrictions(self.restrictions, dict(zip(self.param_names, args)), False)
            parameter_space.addConstraint(restrictions_wrapper, self.param_names)
        elif self.restrictions is not None:
            raise ValueError(f"The restrictions are of unsupported type {type(self.restrictions)}")
        return parameter_space

    def is_param_config_valid(self, param_config: tuple) -> bool:
        """ returns whether the parameter config is valid (i.e. is in the searchspace after restrictions) """
        return self.get_param_config_index(param_config) is not None

    def get_list_dict(self) -> dict:
        """ get the internal dictionary """
        return self.__dict

    def get_param_indices(self, param_config: tuple) -> tuple:
        """ for each parameter value in the param config, find the index in the tunable parameters """
        return tuple(self.params_values[index].index(param_value) for index, param_value in enumerate(param_config))

    def get_param_configs_at_indices(self, indices: List[int]) -> List[tuple]:
        """ Get the param configs at the given indices """
        # map(get) is ~40% faster than numpy[indices] (average based on six searchspaces with 10000, 100000 and 1000000 configs and 10 or 100 random indices)
        return list(map(self.list.__getitem__, indices))

    def get_param_config_index(self, param_config: tuple):
        """ Lookup the index for a parameter configuration, returns None if not found """
        # constant time O(1) access - much faster than any other method, but needs a shadow dict of the search space
        return self.__dict.get(param_config, None)

    def __prepare_neighbors_index(self):
        """ prepare by calculating the indices for the individual parameters """
        self.params_values_indices = np.array(list(self.get_param_indices(param_config) for param_config in self.list))

    def __get_neighbors_indices_hamming(self, param_config: tuple) -> List[int]:
        """ get the neighbors using Hamming distance from the parameter configuration """
        num_matching_params = np.count_nonzero(self.__numpy == param_config, -1)
        matching_indices = (num_matching_params == self.num_params - 1).nonzero()[0]
        return matching_indices

    def __get_neighbors_indices_strictlyadjacent(self, param_config_index: int = None, param_config: tuple = None) -> List[int]:
        """ get the neighbors using strictly adjacent distance from the parameter configuration (parameter index absolute difference == 1) """
        param_config_value_indices = self.get_param_indices(param_config) if param_config_index is None else self.params_values_indices[param_config_index]
        # calculate the absolute difference between the parameter value indices
        abs_index_difference = np.abs(self.params_values_indices - param_config_value_indices)
        # get the param config indices where the difference is one or less for each position
        matching_indices = (np.max(abs_index_difference, axis=1) <= 1).nonzero()[0]
        # as the selected param config does not differ anywhere, remove it from the matches
        if param_config_index is not None:
            matching_indices = np.setdiff1d(matching_indices, [param_config_index], assume_unique=False)
        return matching_indices

    def __get_neighbors_indices_adjacent(self, param_config_index: int = None, param_config: tuple = None) -> List[int]:
        """ get the neighbors using adjacent distance from the parameter configuration (parameter index absolute difference >= 1)"""
        param_config_value_indices = self.get_param_indices(param_config) if param_config_index is None else self.params_values_indices[param_config_index]
        # calculate the difference between the parameter value indices
        index_difference = self.params_values_indices - param_config_value_indices
        # transpose to get the param indices difference per parameter instead of per param config
        index_difference_transposed = index_difference.transpose()
        # for each parameter get the closest upper and lower parameter (absolute index difference >= 1)
        # np.PINF has been replaced by 1e12 here, as on some systems np.PINF becomes np.NINF
        upper_bound = tuple(
            np.min(index_difference_transposed[p][(index_difference_transposed[p] > 0).nonzero()], initial=1e12) for p in range(self.num_params))
        lower_bound = tuple(
            np.max(index_difference_transposed[p][(index_difference_transposed[p] < 0).nonzero()], initial=-1e12) for p in range(self.num_params))
        # return the indices where each parameter is within bounds
        matching_indices = np.logical_and(index_difference <= upper_bound, index_difference >= lower_bound).all(axis=1).nonzero()[0]
        # as the selected param config does not differ anywhere, remove it from the matches
        if param_config_index is not None:
            matching_indices = np.setdiff1d(matching_indices, [param_config_index], assume_unique=False)
        return matching_indices

    def __build_neighbors_index(self, neighbor_method) -> np.ndarray:
        """ build an index of the neighbors for each parameter configuration """
        # for Hamming no preperation is necessary, find the neighboring parameter configurations
        if neighbor_method == 'Hamming':
            return np.array(list(self.__get_neighbors_indices_hamming(param_config) for param_config in self.list))

        # for each parameter configuration, find the neighboring parameter configurations
        if self.params_values_indices is None:
            self.__prepare_neighbors_index()
        if neighbor_method == 'strictly-adjacent':
            return np.array(
                list(
                    self.__get_neighbors_indices_strictlyadjacent(param_config_index, param_config)
                    for param_config_index, param_config in enumerate(self.list)))
        if neighbor_method == 'adjacent':
            return np.array(
                list(self.__get_neighbors_indices_adjacent(param_config_index, param_config) for param_config_index, param_config in enumerate(self.list)))
        raise NotImplementedError()

    def get_random_sample_indices(self, num_samples: int) -> np.ndarray:
        """ Get the list indices for a random, non-conflicting sample """
        if num_samples > self.size:
            raise ValueError("The number of samples requested is greater than the searchspace size")
        return np.random.choice(self.indices, size=num_samples, replace=False)

    def get_random_sample(self, num_samples: int) -> List[tuple]:
        """ Get the parameter configurations for a random, non-conflicting sample (caution: not unique in consecutive calls) """
        return self.get_param_configs_at_indices(self.get_random_sample_indices(num_samples))

    def get_neighbors_indices_no_cache(self, param_config: tuple, neighbor_method=None) -> List[int]:
        """ Get the neighbors indices for a parameter configuration (does not check running cache, useful when mixing neighbor methods) """
        param_config_index = self.get_param_config_index(param_config)

        # this is the simplest case, just return the cached value
        if self.build_neighbors_index and param_config_index is not None:
            if neighbor_method is not None and neighbor_method != self.neighbor_method:
                raise ValueError(f"The neighbor method {neighbor_method} differs from the neighbor method {self.neighbor_method} initially used for indexing")
            return self.neighbors_index[param_config_index]

        # check if the neighbor methods do not differ
        if self.neighbor_method != neighbor_method and self.neighbor_method is not None and neighbor_method is not None:
            raise ValueError(f"The neighbor method {neighbor_method} differs from the intially set {self.neighbor_method}")
        if neighbor_method is None:
            neighbor_method = self.neighbor_method

        if neighbor_method == 'Hamming':
            return self.__get_neighbors_indices_hamming(param_config)

        # prepare the indices if necessary
        if self.params_values_indices is None:
            self.__prepare_neighbors_index()

        # if the passed param_config is fictious, we can not use the pre-calculated neighbors index
        if neighbor_method == 'strictly-adjacent':
            return self.__get_neighbors_indices_strictlyadjacent(param_config_index, param_config)
        if neighbor_method == 'adjacent':
            return self.__get_neighbors_indices_adjacent(param_config_index, param_config)
        raise ValueError(f"The neighbor method {neighbor_method} is not in {supported_neighbor_methods}")

    def get_neighbors_indices(self, param_config: tuple, neighbor_method=None) -> List[int]:
        """ Get the neighbors indices for a parameter configuration, possibly cached """
        neighbors = self.__neighbor_cache.get(param_config, None)
        if neighbors is None:
            neighbors = self.get_neighbors_indices_no_cache(param_config, neighbor_method)
            self.__neighbor_cache[param_config] = neighbors
        return neighbors

    def get_neighbors_no_cache(self, param_config: tuple, neighbor_method=None) -> List[tuple]:
        """ Get the neighbors for a parameter configuration (does not check running cache, useful when mixing neighbor methods) """
        return self.get_param_configs_at_indices(self.get_neighbors_indices_no_cache(param_config, neighbor_method))

    def get_neighbors(self, param_config: tuple, neighbor_method=None) -> List[tuple]:
        """ Get the neighbors for a parameter configuration """
        return self.get_param_configs_at_indices(self.get_neighbors_indices(param_config, neighbor_method))

    def get_param_neighbors(self, param_config: tuple, index: int, neighbor_method: str, randomize: bool) -> list:
        """ Get the neighboring parameters at an index """
        original_value = param_config[index]
        params = list(set(neighbor[index] for neighbor in self.get_neighbors(param_config, neighbor_method) if neighbor[index] != original_value))
        if randomize:
            shuffle(params)
        return params

    def order_param_configs(self, param_configs: List[tuple], order: List[int], randomize_in_params=True) -> List[tuple]:
        """ Order a list of parameter configurations based on the indices of the parameters given, starting at 0. If randomize_params is true, the order within parameters is shuffled. """
        if len(order) != self.num_params:
            raise ValueError(f"The length of the order ({len(order)}) must be equal to the number of parameters ({self.num_params})")
        for i in range(self.num_params):
            if i not in order:
                raise ValueError(f"order needs to be a list of the parameter indices, but index {i} is missing")

        # choose the comparison basis and add it as the first in the order
        base_comparison = choice(param_configs)
        ordered_param_configs = list([base_comparison])

        # move through the parameters in order, if a configuration does not match the base comparison add it to the list
        for param_index in order:
            sublist = list()
            for param_config in param_configs:
                if param_config[param_index] != base_comparison[param_index] and param_config not in ordered_param_configs:
                    ordered_param_configs.append(param_config)
            # randomize the order within the parameters
            if randomize_in_params:
                shuffle(sublist)
            # append to the ordered list
            ordered_param_configs += sublist

        # check that the number of elements still matches before returning
        if len(param_configs) != len(ordered_param_configs):
            raise ValueError(
                f"The number of ordered parameter configurations ({len(ordered_param_configs)}) differs from the original number of parameter configurations ({len(param_configs)})"
            )
        return ordered_param_configs
