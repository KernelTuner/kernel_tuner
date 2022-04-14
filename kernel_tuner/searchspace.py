from typing import Tuple

from util import default_block_size_names
from util import check_restrictions as check_instance_restrictions
from util import MaxProdConstraint

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

        # keep the
        self.list, self.__list = self.__build_searchspace()
        self.size = self.list.size
        self.num_params = len(self.list[0])
        self.indices = np.arange(self.size)
        if build_neighbors_index:
            self.neighbors_index = self.__build_neighbors_index(neighbor_method)

    def __build_searchspace(self) -> Tuple[np.ndarray, tuple]:
        """ compute valid configurations in a search space based on restrictions and max_threads """

        # instantiate the parameter space with all the variables
        parameter_space = Problem()
        for param_name, param_values in self.tune_params.items():
            parameter_space.addVariable(param_name, param_values)

        # add the user-specified restrictions as constraints on the parameter space
        if isinstance(self.restrictions, list):
            for restriction in self.restrictions:
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
        parameter_space_tuple = tuple(parameter_space_list)
        parameter_space_numpy = np.array(parameter_space_list)
        size_unique = np.unique(parameter_space_numpy, axis=-1).size
        if size_unique != parameter_space_numpy.size:
            raise ValueError(f"{parameter_space_numpy.size - size_unique} duplicate parameter configurations in the searchspace, this should not happen")
        return parameter_space_numpy, parameter_space_tuple

    def __prepare_neighbors_index(self):
        """ prepare by calculating the indices for the individual parameters """
        params_values_indices = np.full((self.size, ), [])
        for param_config_index, param_config in enumerate(self.list):
            # for each parameter value in the param config, find the index in the tunable parameters
            params_values_indices[param_config_index] = tuple(self.params_values[index].index(param_value) for index, param_value in enumerate(param_config))
        self.params_values_indices = params_values_indices

    def __build_neighbors_index(self, neighbor_method) -> np.ndarray:
        """ build an index of the neighbors for each parameter configuration """
        neighbors_indices = np.full((self.size, ), [])

        # for Hamming no preperation is necessary, find the neighboring parameter configurations
        if neighbor_method == 'Hamming':
            for param_config_index, param_config in enumerate(self.list):
                num_matching_params = np.count_nonzero(self.list == param_config, -1)
                matching_indices = (num_matching_params == self.num_params - 1)
                neighbors_indices[param_config_index] = matching_indices
            return neighbors_indices

        # for each parameter configuration, find the neighboring parameter configurations
        self.__prepare_neighbors_index()
        if neighbor_method == 'strictly-adjacent':
            # TODO
            return neighbors_indices
        elif neighbor_method == 'adjacent':
            # TODO
            return neighbors_indices
        else:
            raise NotImplementedError()

    def get_param_config_index(self, param_config: tuple):
        """ Lookup the index for a parameter configuration, returns None if not found """
        # two options: use .index() on python list / tuple, or np.count_nonzero + np.nonzero mask

        #1 pros: ~4.5x speedup over numpy method (see test_index_performance.py), cons: need to keep a shadow list of the search space
        try:
            return self.__list.index(param_config)
        except ValueError:
            return None

        #2 pros: pure numpy & no need to keep a shadow list, cons: does not stop on first occurance, much slower
        num_matching_params = np.count_nonzero(self.list == param_config, -1)
        indices = (num_matching_params == self.num_params).nonzero()[0]
        return indices[0] if len(indices) == 1 else None

    def get_random_sample_indices(self, num_samples: int) -> np.ndarray:
        """ Get the list indices for a random, non-conflicting sample """
        return np.random.choice(self.indices, size=num_samples, replace=False)

    def get_random_sample(self, num_samples: int) -> np.ndarray:
        """ Get the parameter configurations for a random, non-conflicting sample """
        return self.list[self.get_random_sample_indices(num_samples)]

    def get_neighbors(self, param_config: tuple, neighbor_method=None) -> np.ndarray:
        """ Get the neighbors for a parameter configuration """
        param_config_index = self.get_param_config_index(param_config)

        # this is the simplest case, just return the cached value
        if self.build_neighbors_index and param_config_index is not None:
            if neighbor_method is not None and neighbor_method != self.neighbor_method:
                raise ValueError(f"The neighbor method {neighbor_method} differs from the neighbor method {self.neighbor_method} initially used for indexing")
            return self.list[self.neighbors_index[param_config_index]]

        # check if the neighbor methods do not differ
        if self.neighbor_method != neighbor_method and self.neighbor_method is not None:
            raise ValueError(f"The neighbor method {neighbor_method} differs from the intially set {self.neighbor_method}")

        # prepare the indices if necessary
        if self.params_values_indices is None and neighbor_method != 'Hamming':
            self.__prepare_neighbors_index()

        # if the passed param_config is fictious, we can not use the pre-calculated neighbors index
        if neighbor_method == 'strictly-adjacent':
            pass
        elif neighbor_method == 'adjacent':
            pass
        elif neighbor_method == 'Hamming':
            pass
        else:
            raise ValueError(f"The neighbor method {neighbor_method} is not in {supported_neighbor_methods}")
