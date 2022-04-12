from util import default_block_size_names
from util import check_restrictions as check_instance_restrictions
from util import MaxProdConstraint

from constraint import Problem, Constraint, FunctionConstraint
import numpy as np


class Searchspace():
    """ Class that offers the search space to strategies """

    def __init__(self, tuning_options: dict, max_threads: int) -> None:
        self.restrictions = tuning_options.restrictions
        self.tune_params = tuning_options.tune_params
        self.param_names = list(self.tune_params.keys())
        self.max_threads = max_threads

        self.list = self.__build_searchspace()
        self.size = self.list.size
        self.indices = np.arange(self.size)

    def __build_searchspace(self) -> np.ndarray:
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
            if param_config not in parameter_space_list:
                parameter_space_list.append(param_config)
            else:
                print(f"Duplicate {param_config}")
        return np.array(parameter_space_list)

    def get_random_sample_indices(self, num_samples: int) -> np.ndarray:
        """ Get the list indices for a random, non-conflicting sample """
        return np.random.choice(self.indices, size=num_samples, replace=False)

    def get_random_sample(self, num_samples: int) -> np.ndarray:
        """ Get the parameter configurations for a random, non-conflicting sample """
        return self.list[self.get_random_sample_indices(num_samples)]
