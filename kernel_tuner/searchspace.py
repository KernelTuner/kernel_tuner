from typing import Tuple, List

from kernel_tuner.util import default_block_size_names
from kernel_tuner.util import check_restrictions as check_instance_restrictions
from kernel_tuner.util import MaxProdConstraint

from constraint import Problem, Constraint, FunctionConstraint
import numpy as np

supported_neighbor_methods = ['strictly-adjacent', 'adjacent', 'Hamming']


class Searchspace():
    """ Class that offers the search space to strategies """

    def __init__(self, tuning_options: dict, max_threads: int, build_neighbors_index=False, neighbor_method=None) -> None:
        """ Build a searchspace using the variables and constraints.
            Optionally build the neighbors index - only faster if you repeatedly look up neighbors. Methods:
                strictly-adjacent: differs +1 or -1 parameter value for each parameter (strictest)
                adjacent: differs the least in each direction for each parameter    (least strict, hardest to implement)
                Hamming: any parameter config with 1 different parameter is a neighbor  (easiest to implement)
        """
        self.tuning_options = tuning_options
        self.restrictions = tuning_options.restrictions
        self.tune_params = tuning_options.tune_params
        self.max_threads = max_threads
        self.param_names = list(self.tune_params.keys())
        self.params_values = tuple(tuple(param_vals) for param_vals in self.tune_params.values())
        self.params_values_indices = None
        self.build_neighbors_index = build_neighbors_index
        self.neighbor_method = neighbor_method
        if (neighbor_method is not None or build_neighbors_index) and neighbor_method not in supported_neighbor_methods:
            raise ValueError(f"Neighbor method is {neighbor_method}, must be one of {supported_neighbor_methods}")

        self.list, self.__numpy, self.__dict, self.size = self.__build_searchspace()
        self.num_params = len(self.list[0])
        self.indices = np.arange(self.size)
        if build_neighbors_index:
            self.neighbors_index = self.__build_neighbors_index(neighbor_method)

    def __build_searchspace(self) -> Tuple[List[tuple], np.ndarray, dict, int]:
        """ compute valid configurations in a search space based on restrictions and max_threads, returns the searchspace, a dict of the searchspace for fast lookups and the size """

        # instantiate the parameter space with all the variables
        parameter_space = Problem()
        for param_name, param_values in self.tune_params.items():
            parameter_space.addVariable(param_name, param_values)

        # add the user-specified restrictions as constraints on the parameter space
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

        # add the default blocksize threads restrictions last, because it is unlikely to reduce the parameter space by much
        block_size_names = self.tuning_options.get("block_size_names", default_block_size_names)
        block_size_names = list(block_size_name for block_size_name in block_size_names if block_size_name in self.param_names)
        if len(block_size_names) > 0:
            parameter_space.addConstraint(MaxProdConstraint(self.max_threads), block_size_names)

        # construct the parameter space with the constraints applied
        parameter_space = parameter_space.getSolutions()

        # form the parameter tuples in the order specified by tune_params.keys()
        parameter_space_list = list()
        for params in parameter_space:
            param_config = tuple(params[param_name] for param_name in self.param_names)
            parameter_space_list.append(param_config)

        # in orded to have the tuples as tuples in numpy, the types are set with a string, but this will make the type np.void
        # type_string = ",".join(list(type(param).__name__ for param in parameter_space_list[0]))
        parameter_space_numpy = np.array(parameter_space_list)
        parameter_space_dict = dict(zip(parameter_space_list, list(range(parameter_space_numpy.size))))

        # check for duplicates
        size_list = len(parameter_space_list)
        size_dict = len(parameter_space_dict.keys())
        if size_list != size_dict:
            raise ValueError(f"{size_list - size_dict} duplicate parameter configurations in the searchspace, this should not happen")

        return parameter_space_list, parameter_space_numpy, parameter_space_dict, size_list

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
        # three options: use .get on python dict, use .index() on python list / tuple, or np.count_nonzero + np.nonzero mask

        #1 pros: constant time O(1) access - much faster than any other method, cons: need to keep a shadow dict of the search space
        return self.__dict.get(param_config, None)

        #2 pros: ~4.5x speedup over numpy method (see test_index_performance.py), cons: need to keep a shadow list of the search space
        try:
            return self.__list.index(param_config)
        except ValueError:
            return None

        #3 pros: pure numpy & no need to keep a shadow list, cons: does not stop on first occurance, much slower
        num_matching_params = np.count_nonzero(self.__numpy == param_config, -1)
        indices = (num_matching_params == self.num_params).nonzero()[0]
        return indices[0] if len(indices) == 1 else None

    def __prepare_neighbors_index(self):
        """ prepare by calculating the indices for the individual parameters """
        self.params_values_indices = np.array(list(self.get_param_indices(param_config) for param_config in self.list))

    def __get_neighbors_indices_hamming(self, param_config: tuple) -> List[int]:
        """ get the neighbors using Hamming distance from the parameter configuration """
        num_matching_params = np.count_nonzero(self.__numpy == param_config, -1)
        matching_indices = (num_matching_params == self.num_params - 1).nonzero()[0]
        return matching_indices

    def __get_neighbors_indices_strictlyadjacent(self, param_config_index: int = None, param_config: tuple = None) -> List[int]:
        """ get the neighbors using strictly adjacent distance from the parameter configuration """
        param_config_value_indices = self.get_param_indices(param_config) if param_config_index is None else self.params_values_indices[param_config_index]
        # calculate the absolute difference between the parameter value indices
        abs_index_difference = np.abs(self.params_values_indices - param_config_value_indices)
        # get the param config indices where the difference is one or less for each position
        matching_indices = (np.max(abs_index_difference, 1) <= 1).nonzero()[0]
        # as the selected param config does not differ anywhere, remove it from the matches
        matching_indices = np.setdiff1d(matching_indices, [param_config_index], assume_unique=False)
        return matching_indices

    def __get_neighbors_indices_adjacent(self, param_config_index: int, param_config: tuple) -> List[int]:
        """ get the neighbors using strictly adjacent distance from the parameter configuration """
        # TODO this is not yet correct, maybe do it in the same style as __get_neighbors_indices_strictlyadjacent except using minimization?
        params_values_indices = self.params_values_indices[param_config_index]
        # for each parameter in the config, take one higher and lower parameter value, if these are valid they are neighbors
        for index, param in param_config:
            param_values = self.params_values[index]
            param_values_index = param_values.index(param)
            upper_params_values_indices = params_values_indices
            lower_params_values_indices = params_values_indices
            upper_params_values_indices[index] = self.params_values[param_values_index + 1] if len(param_values) > param_values_index else None
            lower_params_values_indices[index] = self.params_values[param_values_index - 1] if param_values_index > 0 else None
        raise NotImplementedError()

    def __build_neighbors_index(self, neighbor_method) -> np.ndarray:
        """ build an index of the neighbors for each parameter configuration """
        # for Hamming no preperation is necessary, find the neighboring parameter configurations
        if neighbor_method == 'Hamming':
            return np.array(list(self.__get_neighbors_indices_hamming(param_config) for param_config in self.list))

        # for each parameter configuration, find the neighboring parameter configurations
        self.__prepare_neighbors_index()
        if neighbor_method == 'strictly-adjacent':
            return np.array(list(self.__get_neighbors_indices_strictlyadjacent(param_config_index) for param_config_index in self.indices))
        elif neighbor_method == 'adjacent':
            return np.array(
                list(self.__get_neighbors_indices_adjacent(param_config, param_config_index) for param_config_index, param_config in enumerate(self.list)))
        else:
            raise NotImplementedError()

    def get_random_sample_indices(self, num_samples: int) -> np.ndarray:
        """ Get the list indices for a random, non-conflicting sample """
        if num_samples > self.size:
            raise ValueError(f"The number of samples requested is greater than the searchspace size")
        return np.random.choice(self.indices, size=num_samples, replace=False)

    def get_random_sample(self, num_samples: int) -> List[tuple]:
        """ Get the parameter configurations for a random, non-conflicting sample (caution: not unique in consecutive calls) """
        return self.get_param_configs_at_indices(self.get_random_sample_indices(num_samples))

    def get_neighbors_indices(self, param_config: tuple, neighbor_method=None) -> List[int]:
        """ Get the neighbors indices for a parameter configuration """
        param_config_index = self.get_param_config_index(param_config)

        # this is the simplest case, just return the cached value
        if self.build_neighbors_index and param_config_index is not None:
            if neighbor_method is not None and neighbor_method != self.neighbor_method:
                raise ValueError(f"The neighbor method {neighbor_method} differs from the neighbor method {self.neighbor_method} initially used for indexing")
            return self.neighbors_index[param_config_index]

        # check if the neighbor methods do not differ
        if self.neighbor_method != neighbor_method and self.neighbor_method is not None:
            raise ValueError(f"The neighbor method {neighbor_method} differs from the intially set {self.neighbor_method}")
        elif neighbor_method == None:
            neighbor_method = self.neighbor_method

        if neighbor_method == 'Hamming':
            return self.__get_neighbors_indices_hamming(param_config)

        # prepare the indices if necessary
        if self.params_values_indices is None:
            self.__prepare_neighbors_index()

        # if the passed param_config is fictious, we can not use the pre-calculated neighbors index
        if neighbor_method == 'strictly-adjacent':
            return self.__get_neighbors_indices_strictlyadjacent(param_config_index, param_config)
        elif neighbor_method == 'adjacent':
            return self.__get_neighbors_indices_adjacent(param_config_index, param_config)
        else:
            raise ValueError(f"The neighbor method {neighbor_method} is not in {supported_neighbor_methods}")

    def get_neighbors(self, param_config: tuple, neighbor_method=None) -> List[tuple]:
        """ Get the neighbors for a parameter configuration """
        return self.get_param_configs_at_indices(self.get_neighbors_indices(param_config, neighbor_method))
