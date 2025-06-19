import ast
import numbers
import re
from pathlib import Path
from random import choice, shuffle
from typing import List, Union
from warnings import warn

import numpy as np
from constraint import (
    BacktrackingSolver,
    Constraint,
    FunctionConstraint,
    MaxProdConstraint,
    MinConflictsSolver,
    OptimizedBacktrackingSolver,
    # ParallelSolver,
    Problem,
    RecursiveBacktrackingSolver,
    Solver,
)

try:
    import torch
    from torch import Tensor

    torch_available = True
except ImportError:
    torch_available = False

from kernel_tuner.util import check_restrictions as check_instance_restrictions
from kernel_tuner.util import (
    compile_restrictions,
    default_block_size_names,
    get_interval,
)

supported_neighbor_methods = ["strictly-adjacent", "adjacent", "Hamming"]


class Searchspace:
    """Class that provides the search space to strategies."""

    def __init__(
        self,
        tune_params: dict,
        restrictions,
        max_threads: int,
        block_size_names=default_block_size_names,
        build_neighbors_index=False,
        neighbor_method=None,
        from_cache: dict = None,
        framework="PythonConstraint",
        solver_method="PC_OptimizedBacktrackingSolver",
        path_to_ATF_cache: Path = None,
    ) -> None:
        """Build a searchspace using the variables and constraints.

        Optionally build the neighbors index - only faster if you repeatedly look up neighbors. Methods:
            strictly-adjacent: differs +1 or -1 parameter index value for each parameter
            adjacent: picks closest parameter value in both directions for each parameter
            Hamming: any parameter config with 1 different parameter value is a neighbor
        Optionally sort the searchspace by the order in which the parameter values were specified. By default, sort goes from first to last parameter, to reverse this use sort_last_param_first.
        Optionally an imported cache can be used instead with `from_cache`, in which case the `tune_params`, `restrictions` and `max_threads` arguments can be set to None, and construction is skipped.
        """
        # check the arguments
        if from_cache is not None:
            assert (
                tune_params is None and restrictions is None and max_threads is None
            ), "When `from_cache` is used, the positional arguments must be set to None."
            tune_params = from_cache["tune_params"]
        if from_cache is None:
            assert tune_params is not None and max_threads is not None, "Must specify positional arguments."

        # set the object attributes using the arguments
        framework_l = framework.lower()
        restrictions = restrictions if restrictions is not None else []
        self.tune_params = tune_params
        self.tune_params_pyatf = None
        self._tensorspace = None
        self.tensor_dtype = torch.float32 if torch_available else None
        self.tensor_device = torch.device("cpu") if torch_available else None
        self.tensor_kwargs = dict(dtype=self.tensor_dtype, device=self.tensor_device)
        self._tensorspace_bounds = None
        self._tensorspace_bounds_indices = []
        self._tensorspace_categorical_dimensions = []
        self._tensorspace_param_config_structure = []
        self._map_tensor_to_param = {}
        self._map_param_to_tensor = {}
        self.restrictions = restrictions.copy() if hasattr(restrictions, "copy") else restrictions
        # the searchspace can add commonly used constraints (e.g. maxprod(blocks) <= maxthreads)
        self._modified_restrictions = restrictions.copy() if hasattr(restrictions, "copy") else restrictions
        self.param_names = list(self.tune_params.keys())
        self.params_values = tuple(tuple(param_vals) for param_vals in self.tune_params.values())
        self.params_values_indices = None
        self.build_neighbors_index = build_neighbors_index
        self.solver_method = solver_method
        self.__neighbor_cache = dict()
        self.neighbor_method = neighbor_method
        if (neighbor_method is not None or build_neighbors_index) and neighbor_method not in supported_neighbor_methods:
            raise ValueError(f"Neighbor method is {neighbor_method}, must be one of {supported_neighbor_methods}")

        # if there are strings in the restrictions, parse them to split constraints or functions (improves solver performance)
        restrictions = [restrictions] if not isinstance(restrictions, list) else restrictions
        if (
            len(restrictions) > 0
            and (
                any(isinstance(restriction, str) for restriction in restrictions)
                or any(
                    isinstance(restriction[0], str) for restriction in restrictions if isinstance(restriction, tuple)
                )
            )
            and not (
                framework_l == "pysmt" or framework_l == "bruteforce" or solver_method.lower() == "pc_parallelsolver"
            )
        ):
            self.restrictions = compile_restrictions(
                restrictions,
                tune_params,
                monolithic=False,
                format=framework_l if framework_l == "pyatf" else None,
                try_to_constraint=framework_l == "pythonconstraint",
            )

        # if an imported cache, skip building and set the values directly
        if from_cache is not None:
            configs = dict(from_cache["cache"]).values()
            self.list = list(tuple([v for p, v in c.items() if p in self.tune_params]) for c in configs)
            self.size = len(self.list)
            self.__dict = dict(zip(self.list, range(self.size)))
        else:
            # get the framework given the framework argument
            if framework_l == "pythonconstraint":
                searchspace_builder = self.__build_searchspace
            elif framework_l == "pysmt":
                searchspace_builder = self.__build_searchspace_pysmt
            elif framework_l == "pyatf":
                searchspace_builder = self.__build_searchspace_pyATF
            elif framework_l == "atf_cache":
                searchspace_builder = self.__build_searchspace_ATF_cache
                self.path_to_ATF_cache = path_to_ATF_cache
            elif framework_l == "bruteforce":
                searchspace_builder = self.__build_searchspace_bruteforce
            else:
                raise ValueError(f"Invalid framework parameter {framework}")

            # get the solver given the solver method argument
            solver = ""
            if solver_method.lower() == "pc_backtrackingsolver":
                solver = BacktrackingSolver()
            elif solver_method.lower() == "pc_optimizedbacktrackingsolver":
                solver = OptimizedBacktrackingSolver(forwardcheck=False)
            elif solver_method.lower() == "pc_parallelsolver":
                raise NotImplementedError("ParallelSolver is not yet implemented")
                # solver = ParallelSolver()
            elif solver_method.lower() == "pc_recursivebacktrackingsolver":
                solver = RecursiveBacktrackingSolver()
            elif solver_method.lower() == "pc_minconflictssolver":
                solver = MinConflictsSolver()
            else:
                raise ValueError(f"Solver method {solver_method} not recognized.")

            # build the search space
            self.list, self.__dict, self.size = searchspace_builder(block_size_names, max_threads, solver)

        # finalize construction
        self.__numpy = None
        self.num_params = len(self.tune_params)
        self.indices = np.arange(self.size)
        if neighbor_method is not None and neighbor_method != "Hamming":
            self.__prepare_neighbors_index()
        if build_neighbors_index:
            self.neighbors_index = self.__build_neighbors_index(neighbor_method)

    # def __build_searchspace_ortools(self, block_size_names: list, max_threads: int) -> Tuple[List[tuple], np.ndarray, dict, int]:
    #     # Based on https://developers.google.com/optimization/cp/cp_solver#python_2
    #     from ortools.sat.python import cp_model

    #     # instantiate the parameter space with all the variables
    #     parameter_space = cp_model.CpModel()
    #     for param_name, param_values in self.tune_params.items():
    #         parameter_space.NewIntervalVar(min(param_values), )
    #         parameter_space.addVariable(param_name, param_values)

    # def __build_searchspace_cpmpy():
    #     # Based on examples in https://github.com/CPMpy/cpmpy/blob/master/examples/nqueens_1000.ipynb
    #     # possible solution for interrupted ranges with 'conso' in https://github.com/CPMpy/cpmpy/blob/master/examples/mario.py
    #     import cpmpy

    #     cpmpy.intvar()

    # def __build_searchspace_pycsp(self, block_size_names: list, max_threads: int):
    #     import pycsp3 as csp

    #     # instantiate the parameter space with all the variables
    #     vars_and_constraints = list()
    #     for param_name, param_values in self.tune_params.items():
    #         var = csp.Var(param_values, id=param_name)
    #         vars_and_constraints.append(var)

    #     # construct the parameter space with the constraints applied
    #     csp.satisfy(*vars_and_constraints)

    #     # solve for all configurations to get the feasible region
    #     if csp.solve(sols=csp.ALL) is csp.SAT:
    #         num_solutions: int = csp.n_solutions()  # number of solutions
    #         solutions = [csp.values(sol=i) for i in range(num_solutions)]  # list of solutions

    def __build_searchspace_bruteforce(self, block_size_names: list, max_threads: int, solver=None):
        # bruteforce solving of the searchspace

        from itertools import product

        from kernel_tuner.util import check_restrictions

        tune_params = self.tune_params
        restrictions = self.restrictions

        # compute cartesian product of all tunable parameters
        parameter_space = product(*tune_params.values())

        # check if there are block sizes in the parameters, if so add default restrictions
        used_block_size_names = list(
            block_size_name for block_size_name in default_block_size_names if block_size_name in tune_params
        )
        if len(used_block_size_names) > 0:
            if not isinstance(restrictions, list):
                restrictions = [restrictions]
            block_size_restriction_spaced = f"{' * '.join(used_block_size_names)} <= {max_threads}"
            block_size_restriction_unspaced = f"{'*'.join(used_block_size_names)} <= {max_threads}"
            if (
                block_size_restriction_spaced not in restrictions
                and block_size_restriction_unspaced not in restrictions
            ):
                restrictions.append(block_size_restriction_spaced)
                if (
                    isinstance(self._modified_restrictions, list)
                    and block_size_restriction_spaced not in self._modified_restrictions
                ):
                    print(f"added default block size restriction '{block_size_restriction_spaced}'")
                    self._modified_restrictions.append(block_size_restriction_spaced)
                    if isinstance(self.restrictions, list):
                        self.restrictions.append(block_size_restriction_spaced)

        # check for search space restrictions
        if restrictions is not None:
            parameter_space = filter(
                lambda p: check_restrictions(restrictions, dict(zip(tune_params.keys(), p)), False), parameter_space
            )

        # evaluate to a list
        parameter_space = list(parameter_space)

        # return the results
        return self.__parameter_space_list_to_lookup_and_return_type(parameter_space)

    def __build_searchspace_pysmt(self, block_size_names: list, max_threads: int, solver: Solver):
        # PySMT imports
        from pysmt.oracles import get_logic
        from pysmt.shortcuts import And, Equals, EqualsOrIff, Not, Or, Real, Symbol
        from pysmt.shortcuts import Solver as PySMTSolver
        from pysmt.typing import REAL

        tune_params = self.tune_params
        restrictions = self.restrictions

        # TODO implement block_size_names, max_threads

        def all_smt(formula, keys) -> list:
            target_logic = get_logic(formula)
            partial_models = list()
            with PySMTSolver(logic=target_logic) as solver:
                solver.add_assertion(formula)
                while solver.solve():
                    partial_model = [EqualsOrIff(k, solver.get_value(k)) for k in keys]
                    assertion = Not(And(partial_model))
                    solver.add_assertion(assertion)
                    partial_models.append(partial_model)
            return partial_models

        # setup each tunable parameter
        symbols = dict([(v, Symbol(v, REAL)) for v in tune_params.keys()])
        # symbols = [Symbol(v, REAL) for v in tune_params.keys()]

        # for each tunable parameter, set the list of allowed values
        domains = list()
        for tune_param_key, tune_param_values in tune_params.items():
            domain = Or([Equals(symbols[tune_param_key], Real(float(val))) for val in tune_param_values])
            domains.append(domain)
        domains = And(domains)

        # add the restrictions
        problem = self.__parse_restrictions_pysmt(restrictions, tune_params, symbols)

        # combine the domain and restrictions
        formula = And(domains, problem)

        # get all solutions
        keys = list(symbols.values())
        all_solutions = all_smt(formula, keys)

        # get the values for the parameters
        parameter_space_list = list()
        for solution in all_solutions:
            sol_dict = dict()
            for param in solution:
                param = str(param.serialize()).replace("(", "").replace(")", "")
                key, value = param.split(" = ")
                try:
                    value = ast.literal_eval(value)
                except ValueError:
                    try:
                        value = eval(value)
                    except NameError:
                        pass
                sol_dict[key] = value
            parameter_space_list.append(tuple(sol_dict[param_name] for param_name in list(tune_params.keys())))

        return self.__parameter_space_list_to_lookup_and_return_type(parameter_space_list)

    def __build_searchspace_pyATF(self, block_size_names: list, max_threads: int, solver: Solver):
        """Builds the searchspace using pyATF."""
        from pyatf import TP, Interval, Set, Tuner
        from pyatf.cost_functions.generic import CostFunction
        from pyatf.search_techniques import Exhaustive

        # Define a bogus cost function
        costfunc = CostFunction(":")  # bash no-op

        # add the Kernel Tuner default blocksize threads restrictions
        assert isinstance(self.restrictions, list)
        valid_block_size_names = list(
            block_size_name for block_size_name in block_size_names if block_size_name in self.param_names
        )
        if len(valid_block_size_names) > 0:
            # adding the default blocksize restriction requires recompilation because pyATF requires combined restrictions for the same parameter
            max_block_size_product = f"{' * '.join(valid_block_size_names)} <= {max_threads}"
            restrictions = self._modified_restrictions.copy() + [max_block_size_product]
            self.restrictions = compile_restrictions(
                restrictions, self.tune_params, format="pyatf", try_to_constraint=False
            )

        # build a dictionary of the restrictions, combined based on last parameter
        res_dict = dict()
        registered_params = list()
        registered_restrictions = list()
        for param in self.tune_params.keys():
            registered_params.append(param)
            for index, (res, params, source) in enumerate(self.restrictions):
                if index in registered_restrictions:
                    continue
                if all(p in registered_params for p in params):
                    if param in res_dict:
                        raise KeyError(
                            f"`{param}` is already in res_dict with `{res_dict[param][1]}`, can't add `{source}`"
                        )
                    res_dict[param] = (res, source)
                    print(source, res, param, params)
                    registered_restrictions.append(index)

        # define the Tunable Parameters
        def get_params():
            params = list()
            for index, (key, values) in enumerate(self.tune_params.items()):
                vi = get_interval(values)
                vals = (
                    Interval(vi[0], vi[1], vi[2]) if vi is not None and vi[2] != 0 else Set(*np.array(values).flatten())
                )
                constraint = res_dict.get(key, None)
                constraint_source = None
                if constraint is not None:
                    constraint, constraint_source = constraint
                # in case of a leftover monolithic restriction, append at the last parameter
                if index == len(self.tune_params) - 1 and len(res_dict) == 0 and len(self.restrictions) == 1:
                    res, params, source = self.restrictions[0]
                    assert callable(res)
                    constraint = res
                params.append(TP(key, vals, constraint, constraint_source))
            return params
        
        # set data
        self.tune_params_pyatf = get_params()

        # tune
        _, _, tuning_data = (
            Tuner().verbosity(0).tuning_parameters(*self.tune_params_pyatf).search_technique(Exhaustive()).tune(costfunc)
        )

        # transform the result into a list of parameter configurations for validation
        tune_params = self.tune_params
        parameter_tuple_list = list()
        for entry in tuning_data.history._entries:
            parameter_tuple_list.append(tuple(entry.configuration[p] for p in tune_params.keys()))
        pl = self.__parameter_space_list_to_lookup_and_return_type(parameter_tuple_list)
        return pl

    def __build_searchspace_ATF_cache(self, block_size_names: list, max_threads: int, solver: Solver):
        """Imports the valid configurations from an ATF CSV file, returns the searchspace, a dict of the searchspace for fast lookups and the size."""
        if block_size_names != default_block_size_names or max_threads != 1024:
            raise ValueError(
                "It is not possible to change 'block_size_names' or 'max_threads here, because at this point ATF has already ran.'"
            )
        import pandas as pd

        try:
            df = pd.read_csv(self.path_to_ATF_cache, sep=";")
            list_of_tuples_of_parameters = list(zip(*(df[column] for column in self.param_names)))
        except pd.errors.EmptyDataError:
            list_of_tuples_of_parameters = list()
        return self.__parameter_space_list_to_lookup_and_return_type(list_of_tuples_of_parameters)

    def __parameter_space_list_to_lookup_and_return_type(
        self, parameter_space_list: list[tuple], validate=True
    ) -> tuple[list[tuple], dict[tuple, int], int]:
        """Returns a tuple of the searchspace as a list of tuples, a dict of the searchspace for fast lookups and the size."""
        parameter_space_dict = dict(zip(parameter_space_list, range(len(parameter_space_list))))
        if validate:
            # check for duplicates
            size_list = len(parameter_space_list)
            size_dict = len(parameter_space_dict.keys())
            if size_list != size_dict:
                raise ValueError(
                    f"{size_list - size_dict} duplicate parameter configurations in the searchspace, this should not happen."
                )
        return (
            parameter_space_list,
            parameter_space_dict,
            size_list,
        )

    def __build_searchspace(self, block_size_names: list, max_threads: int, solver: Solver):
        """Compute valid configurations in a search space based on restrictions and max_threads."""
        # instantiate the parameter space with all the variables
        parameter_space = Problem(solver=solver)
        for param_name, param_values in self.tune_params.items():
            parameter_space.addVariable(str(param_name), param_values)

        # add the user-specified restrictions as constraints on the parameter space
        parameter_space = self.__add_restrictions(parameter_space)

        # add the default blocksize threads restrictions last, because it is unlikely to reduce the parameter space by much
        valid_block_size_names = list(
            block_size_name for block_size_name in block_size_names if block_size_name in self.param_names
        )
        if len(valid_block_size_names) > 0:
            parameter_space.addConstraint(MaxProdConstraint(max_threads), valid_block_size_names)
            max_block_size_product = f"{' * '.join(valid_block_size_names)} <= {max_threads}"
            if (
                isinstance(self._modified_restrictions, list)
                and max_block_size_product not in self._modified_restrictions
            ):
                self._modified_restrictions.append(max_block_size_product)
                if isinstance(self.restrictions, list):
                    self.restrictions.append((MaxProdConstraint(max_threads), valid_block_size_names, None))

        # construct the parameter space with the constraints applied
        return parameter_space.getSolutionsAsListDict(order=self.param_names)

    def __add_restrictions(self, parameter_space: Problem) -> Problem:
        """Add the user-specified restrictions as constraints on the parameter space."""
        if isinstance(self.restrictions, list):
            for restriction in self.restrictions:
                required_params = self.param_names

                # convert to a Constraint type if necessary
                if isinstance(restriction, tuple):
                    restriction, required_params, _ = restriction
                if callable(restriction) and not isinstance(restriction, Constraint):
                    restriction = FunctionConstraint(restriction)

                # add the Constraint
                if isinstance(restriction, FunctionConstraint):
                    parameter_space.addConstraint(restriction, required_params)
                elif isinstance(restriction, Constraint):
                    all_params_required = all(param_name in required_params for param_name in self.param_names)
                    parameter_space.addConstraint(restriction, None if all_params_required else required_params)
                elif isinstance(restriction, str) and self.solver_method.lower() == "pc_parallelsolver":
                    parameter_space.addConstraint(restriction)
                else:
                    raise ValueError(f"Unrecognized restriction type {type(restriction)} ({restriction})")

        # if the restrictions are the old monolithic function, apply them directly (only for backwards compatibility, likely slower than well-specified constraints!)
        elif callable(self.restrictions):

            def restrictions_wrapper(*args):
                return check_instance_restrictions(self.restrictions, dict(zip(self.param_names, args)), False)

            parameter_space.addConstraint(FunctionConstraint(restrictions_wrapper), self.param_names)
        elif self.restrictions is not None:
            raise ValueError(f"The restrictions are of unsupported type {type(self.restrictions)}")
        return parameter_space

    def __parse_restrictions_pysmt(self, restrictions: list, tune_params: dict, symbols: dict):
        """Parses restrictions from a list of strings into PySMT compatible restrictions."""
        from pysmt.shortcuts import (
            GE,
            GT,
            LE,
            LT,
            And,
            Bool,
            Div,
            Equals,
            Int,
            Minus,
            Or,
            Plus,
            Pow,
            Real,
            String,
            Times,
        )

        regex_match_variable = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"

        boolean_comparison_mapping = {
            "==": Equals,
            "<": LT,
            "<=": LE,
            ">=": GE,
            ">": GT,
            "&&": And,
            "||": Or,
        }

        operators_mapping = {"+": Plus, "-": Minus, "*": Times, "/": Div, "^": Pow}

        constant_init_mapping = {
            "int": Int,
            "float": Real,
            "str": String,
            "bool": Bool,
        }

        def replace_params(match_object):
            key = match_object.group(1)
            if key in tune_params:
                return 'params["' + key + '"]'
            else:
                return key

        # rewrite the restrictions so variables are singled out
        parsed = [re.sub(regex_match_variable, replace_params, res) for res in restrictions]
        # ensure no duplicates are in the list
        parsed = list(set(parsed))
        # replace ' or ' and ' and ' with ' || ' and ' && '
        parsed = list(r.replace(" or ", " || ").replace(" and ", " && ") for r in parsed)

        # compile each restriction by replacing parameters and operators with their PySMT equivalent
        compiled_restrictions = list()
        for parsed_restriction in parsed:
            words = parsed_restriction.split(" ")

            # make a forward pass over all the words to organize and substitute
            add_next_var_or_constant = False
            var_or_constant_backlog = list()
            operator_backlog = list()
            operator_backlog_left_right = list()
            boolean_backlog = list()
            for word in words:
                if word.startswith("params["):
                    # if variable
                    varname = word.replace('params["', "").replace('"]', "")
                    var = symbols[varname]
                    var_or_constant_backlog.append(var)
                elif word in boolean_comparison_mapping:
                    # if comparator
                    boolean_backlog.append(boolean_comparison_mapping[word])
                    continue
                elif word in operators_mapping:
                    # if operator
                    operator_backlog.append(operators_mapping[word])
                    add_next_var_or_constant = True
                    continue
                else:
                    # if constant: evaluate to check if it is an integer, float, etc. If not, treat it as a string.
                    try:
                        constant = ast.literal_eval(word)
                    except ValueError:
                        constant = word
                    # convert from Python type to PySMT equivalent
                    type_instance = constant_init_mapping[type(constant).__name__]
                    var_or_constant_backlog.append(type_instance(constant))
                if add_next_var_or_constant:
                    right, left = var_or_constant_backlog.pop(-1), var_or_constant_backlog.pop(-1)
                    operator_backlog_left_right.append((left, right, len(var_or_constant_backlog)))
                    add_next_var_or_constant = False
                    # reserve an empty spot for the combined operation to preserve the order
                    var_or_constant_backlog.append(None)

            # for each of the operators, instantiate them with variables or constants
            for i, operator in enumerate(operator_backlog):
                # merges the first two symbols in the backlog into one
                left, right, new_index = operator_backlog_left_right[i]
                assert (
                    var_or_constant_backlog[new_index] is None
                )  # make sure that this is a reserved spot to avoid changing the order
                var_or_constant_backlog[new_index] = operator(left, right)

            # for each of the booleans, instantiate them with variables or constants
            compiled = list()
            assert len(boolean_backlog) <= 1, "Max. one boolean operator per restriction."
            for boolean in boolean_backlog:
                left, right = var_or_constant_backlog.pop(0), var_or_constant_backlog.pop(0)
                compiled.append(boolean(left, right))

            # add the restriction to the list of restrictions
            compiled_restrictions.append(compiled[0])

        return And(compiled_restrictions)

    def sorted_list(self, sort_last_param_first=False):
        """Returns list of parameter configs sorted based on the order in which the parameter values were specified.

        :param sort_last_param_first: By default, sort goes from first to last parameter, to reverse this use sort_last_param_first
        """
        params_values_indices = list(self.get_param_indices(param_config) for param_config in self.list)
        params_values_indices_dict = dict(zip(params_values_indices, list(range(len(params_values_indices)))))

        # Python's built-in sort will sort starting in front, so if we want to vary the first parameter the tuple needs to be reversed
        if sort_last_param_first:
            params_values_indices.sort(key=lambda t: tuple(reversed(t)))
        else:
            params_values_indices.sort()

        # find the index of the parameter configuration for each parameter value index, using a dict to do it in constant time
        new_order = [
            params_values_indices_dict.get(param_values_indices) for param_values_indices in params_values_indices
        ]

        # apply the new order
        return [self.list[i] for i in new_order]

    def is_param_config_valid(self, param_config: tuple) -> bool:
        """Returns whether the parameter config is valid (i.e. is in the searchspace after restrictions)."""
        return self.get_param_config_index(param_config) is not None

    def get_list_dict(self) -> dict:
        """Get the internal dictionary."""
        return self.__dict

    def get_list_numpy(self) -> np.ndarray:
        """Get the parameter space list as a NumPy array. Initializes the NumPy array if not yet done.

        Returns:
            the NumPy array.
        """
        if self.__numpy is None:
            # create a numpy array of the search space
            # in order to have the tuples as tuples in numpy, the types are set with a string, but this will make the type np.void
            # type_string = ",".join(list(type(param).__name__ for param in parameter_space_list[0]))
            self.__numpy = np.array(self.list)
        return self.__numpy

    def get_param_indices(self, param_config: tuple) -> tuple:
        """For each parameter value in the param config, find the index in the tunable parameters."""
        return tuple(self.params_values[index].index(param_value) for index, param_value in enumerate(param_config))

    def get_param_configs_at_indices(self, indices: List[int]) -> List[tuple]:
        """Get the param configs at the given indices."""
        # map(get) is ~40% faster than numpy[indices] (average based on six searchspaces with 10000, 100000 and 1000000 configs and 10 or 100 random indices)
        return list(map(self.list.__getitem__, indices))

    def get_param_config_index(self, param_config: Union[tuple, any]):
        """Lookup the index for a parameter configuration, returns None if not found."""
        if torch_available and isinstance(param_config, Tensor):
            param_config = self.tensor_to_param_config(param_config)
        # constant time O(1) access - much faster than any other method, but needs a shadow dict of the search space
        return self.__dict.get(param_config, None)

    def initialize_tensorspace(self, dtype=None, device=None):
        """Encode the searchspace in a Tensor. Save the mapping. Call this function directly to control the precision or device used."""
        assert self._tensorspace is None, "Tensorspace is already initialized"
        skipped_count = 0
        bounds = []
        if dtype is not None:
            self.tensor_dtype = dtype
        if device is not None:
            self.tensor_device = device
        self.tensor_kwargs = dict(dtype=self.tensor_dtype, device=self.tensor_device)

        # generate the mappings to and from tensor values
        for index, param_values in enumerate(self.params_values):
            # filter out parameters that do not matter, more efficient and avoids bounds problem
            if len(param_values) < 2 or all(p == param_values[0] for p in param_values):
                # keep track of skipped parameters, add them back in conversion functions
                self._tensorspace_param_config_structure.append(param_values[0])
                skipped_count += 1
                continue
            else:
                self._tensorspace_param_config_structure.append(None)

            # convert numericals to float, or encode categorical
            if all(isinstance(v, numbers.Real) for v in param_values):
                tensor_values = torch.tensor(param_values, dtype=self.tensor_dtype)
            else:
                self._tensorspace_categorical_dimensions.append(index - skipped_count)
                # tensor_values = np.arange(len(param_values))
                tensor_values = torch.arange(len(param_values), dtype=self.tensor_dtype)

            # write the mappings to the object
            self._map_param_to_tensor[index] = dict(zip(param_values, tensor_values.tolist()))
            self._map_tensor_to_param[index] = dict(zip(tensor_values.tolist(), param_values))
            bounds.append((tensor_values.min(), tensor_values.max()))
            if tensor_values.min() < tensor_values.max():
                self._tensorspace_bounds_indices.append(index - skipped_count)

        # do some checks
        assert len(self.params_values) == len(self._tensorspace_param_config_structure)
        assert len(self._map_param_to_tensor) == len(self._map_tensor_to_param) == len(bounds)
        assert len(self._tensorspace_bounds_indices) <= len(bounds)

        # apply the mappings on the full searchspace
        # numpy_repr = self.get_list_numpy()
        # numpy_repr = np.apply_along_axis(self.param_config_to_tensor, 1, numpy_repr)
        # self._tensorspace = torch.from_numpy(numpy_repr.astype(self.tensor_dtype)).to(self.tensor_device)
        self._tensorspace = torch.stack(tuple(map(self.param_config_to_tensor, self.list)))

        # set the bounds in the correct format (one array for the min, one for the max)
        bounds = torch.tensor(bounds, **self.tensor_kwargs)
        self._tensorspace_bounds = torch.cat([bounds[:, 0], bounds[:, 1]]).reshape((2, bounds.shape[0]))

    def get_tensorspace(self):
        """Get the searchspace encoded in a Tensor. To use a non-default dtype or device, call `initialize_tensorspace` first."""
        if self._tensorspace is None:
            self.initialize_tensorspace()
        return self._tensorspace

    def get_tensorspace_categorical_dimensions(self):
        """Get the a list of the categorical dimensions in the tensorspace."""
        return self._tensorspace_categorical_dimensions

    def param_config_to_tensor(self, param_config: tuple):
        """Convert from a parameter configuration to a Tensor."""
        if len(self._map_param_to_tensor) == 0:
            self.initialize_tensorspace()
        array = []
        for i, param in enumerate(param_config):
            if self._tensorspace_param_config_structure[i] is not None:
                continue  # skip over parameters not in the tensorspace
            mapping = self._map_param_to_tensor[i]
            conversions = [None, str, float, int, bool]
            for c in conversions:
                try:
                    c_param = param if c is None else c(param)
                    array.append(mapping[c_param])
                    break
                except (KeyError, ValueError) as e:
                    if c == conversions[-1]:
                        raise KeyError(f"No variant of {param} could be found in {mapping}") from e
        return torch.tensor(array, **self.tensor_kwargs)

    def tensor_to_param_config(self, tensor):
        """Convert from a Tensor to a parameter configuration."""
        assert tensor.dim() == 1, f"Parameter configuration tensor must be 1-dimensional, is {tensor.dim()} ({tensor})"
        if len(self._map_tensor_to_param) == 0:
            self.initialize_tensorspace()
        config = self._tensorspace_param_config_structure.copy()
        skip_counter = 0
        for i, param in enumerate(config):
            if param is not None:
                skip_counter += 1
            else:
                value = tensor[i - skip_counter].item()
                config[i] = self._map_tensor_to_param[i][value]
        return tuple(config)

    def get_tensorspace_bounds(self):
        """Get the bounds to the tensorspace parameters, returned as a 2 x d dimensional tensor, and the indices of the parameters."""
        if self._tensorspace is None:
            self.initialize_tensorspace()
        return self._tensorspace_bounds, self._tensorspace_bounds_indices

    def __prepare_neighbors_index(self):
        """Prepare by calculating the indices for the individual parameters."""
        self.params_values_indices = np.array(list(self.get_param_indices(param_config) for param_config in self.list))

    def __get_neighbors_indices_hamming(self, param_config: tuple) -> List[int]:
        """Get the neighbors using Hamming distance from the parameter configuration."""
        num_matching_params = np.count_nonzero(self.get_list_numpy() == param_config, -1)
        matching_indices = (num_matching_params == self.num_params - 1).nonzero()[0]
        return matching_indices

    def __get_neighbors_indices_strictlyadjacent(
        self, param_config_index: int = None, param_config: tuple = None
    ) -> List[int]:
        """Get the neighbors using strictly adjacent distance from the parameter configuration (parameter index absolute difference == 1)."""
        param_config_value_indices = (
            self.get_param_indices(param_config)
            if param_config_index is None
            else self.params_values_indices[param_config_index]
        )
        # calculate the absolute difference between the parameter value indices
        abs_index_difference = np.abs(self.params_values_indices - param_config_value_indices)
        # get the param config indices where the difference is one or less for each position
        matching_indices = (np.max(abs_index_difference, axis=1) <= 1).nonzero()[0]
        # as the selected param config does not differ anywhere, remove it from the matches
        if param_config_index is not None:
            matching_indices = np.setdiff1d(matching_indices, [param_config_index], assume_unique=False)
        return matching_indices

    def __get_neighbors_indices_adjacent(self, param_config_index: int = None, param_config: tuple = None) -> List[int]:
        """Get the neighbors using adjacent distance from the parameter configuration (parameter index absolute difference >= 1)."""
        param_config_value_indices = (
            self.get_param_indices(param_config)
            if param_config_index is None
            else self.params_values_indices[param_config_index]
        )
        # calculate the difference between the parameter value indices
        index_difference = self.params_values_indices - param_config_value_indices
        # transpose to get the param indices difference per parameter instead of per param config
        index_difference_transposed = index_difference.transpose()
        # for each parameter get the closest upper and lower parameter (absolute index difference >= 1)
        # np.PINF has been replaced by 1e12 here, as on some systems np.PINF becomes np.NINF
        upper_bound = tuple(
            np.min(
                index_difference_transposed[p][(index_difference_transposed[p] > 0).nonzero()],
                initial=1e12,
            )
            for p in range(self.num_params)
        )
        lower_bound = tuple(
            np.max(
                index_difference_transposed[p][(index_difference_transposed[p] < 0).nonzero()],
                initial=-1e12,
            )
            for p in range(self.num_params)
        )
        # return the indices where each parameter is within bounds
        matching_indices = (
            np.logical_and(index_difference <= upper_bound, index_difference >= lower_bound).all(axis=1).nonzero()[0]
        )
        # as the selected param config does not differ anywhere, remove it from the matches
        if param_config_index is not None:
            matching_indices = np.setdiff1d(matching_indices, [param_config_index], assume_unique=False)
        return matching_indices

    def __build_neighbors_index(self, neighbor_method) -> List[List[int]]:
        """Build an index of the neighbors for each parameter configuration."""
        # for Hamming no preperation is necessary, find the neighboring parameter configurations
        if neighbor_method == "Hamming":
            return list(self.__get_neighbors_indices_hamming(param_config) for param_config in self.list)

        # for each parameter configuration, find the neighboring parameter configurations
        if self.params_values_indices is None:
            self.__prepare_neighbors_index()
        if neighbor_method == "strictly-adjacent":
            return list(
                self.__get_neighbors_indices_strictlyadjacent(param_config_index, param_config)
                for param_config_index, param_config in enumerate(self.list)
            )

        if neighbor_method == "adjacent":
            return list(
                self.__get_neighbors_indices_adjacent(param_config_index, param_config)
                for param_config_index, param_config in enumerate(self.list)
            )

        raise NotImplementedError(f"The neighbor method {neighbor_method} is not implemented")

    def get_random_sample_indices(self, num_samples: int) -> np.ndarray:
        """Get the list indices for a random, non-conflicting sample."""
        if num_samples > self.size:
            raise ValueError(
                f"The number of samples requested ({num_samples}) is greater than the searchspace size ({self.size})"
            )
        return np.random.choice(self.indices, size=num_samples, replace=False)

    def get_random_sample(self, num_samples: int) -> List[tuple]:
        """Get the parameter configurations for a random, non-conflicting sample (caution: not unique in consecutive calls)."""
        if self.size < num_samples:
            warn(
                f"Too many samples requested ({num_samples}), reducing the number of samples to the searchspace size ({self.size})"
            )
            num_samples = self.size
        return self.get_param_configs_at_indices(self.get_random_sample_indices(num_samples))

    def get_neighbors_indices_no_cache(self, param_config: tuple, neighbor_method=None) -> List[int]:
        """Get the neighbors indices for a parameter configuration (does not check running cache, useful when mixing neighbor methods)."""
        param_config_index = self.get_param_config_index(param_config)

        # this is the simplest case, just return the cached value
        if self.build_neighbors_index and param_config_index is not None:
            if neighbor_method is not None and neighbor_method != self.neighbor_method:
                raise ValueError(
                    f"The neighbor method {neighbor_method} differs from the neighbor method {self.neighbor_method} initially used for indexing"
                )
            return self.neighbors_index[param_config_index]

        # check if there is a neighbor method to use
        if neighbor_method is None:
            if self.neighbor_method is None:
                raise ValueError("Neither the neighbor_method argument nor self.neighbor_method was set")
            neighbor_method = self.neighbor_method

        if neighbor_method == "Hamming":
            return self.__get_neighbors_indices_hamming(param_config)

        # prepare the indices if necessary
        if self.params_values_indices is None:
            self.__prepare_neighbors_index()

        # if the passed param_config is fictious, we can not use the pre-calculated neighbors index
        if neighbor_method == "strictly-adjacent":
            return self.__get_neighbors_indices_strictlyadjacent(param_config_index, param_config)
        if neighbor_method == "adjacent":
            return self.__get_neighbors_indices_adjacent(param_config_index, param_config)
        raise ValueError(f"The neighbor method {neighbor_method} is not in {supported_neighbor_methods}")

    def get_neighbors_indices(self, param_config: tuple, neighbor_method=None) -> List[int]:
        """Get the neighbors indices for a parameter configuration, possibly cached."""
        neighbors = self.__neighbor_cache.get(param_config, None)
        # if there are no cached neighbors, compute them
        if neighbors is None:
            neighbors = self.get_neighbors_indices_no_cache(param_config, neighbor_method)
            self.__neighbor_cache[param_config] = neighbors
        # if the neighbors were cached but the specified neighbor method was different than the one initially used to build the cache, throw an error
        elif (
            self.neighbor_method is not None and neighbor_method is not None and self.neighbor_method != neighbor_method
        ):
            raise ValueError(
                f"The neighbor method {neighbor_method} differs from the intially set {self.neighbor_method}, can not use cached neighbors. Use 'get_neighbors_no_cache()' when mixing neighbor methods to avoid this."
            )
        return neighbors

    def are_neighbors_indices_cached(self, param_config: tuple) -> bool:
        """Returns true if the neighbor indices are in the cache, false otherwise."""
        return param_config in self.__neighbor_cache

    def get_neighbors_no_cache(self, param_config: tuple, neighbor_method=None) -> List[tuple]:
        """Get the neighbors for a parameter configuration (does not check running cache, useful when mixing neighbor methods)."""
        return self.get_param_configs_at_indices(self.get_neighbors_indices_no_cache(param_config, neighbor_method))

    def get_neighbors(self, param_config: tuple, neighbor_method=None) -> List[tuple]:
        """Get the neighbors for a parameter configuration."""
        return self.get_param_configs_at_indices(self.get_neighbors_indices(param_config, neighbor_method))

    def get_param_neighbors(self, param_config: tuple, index: int, neighbor_method: str, randomize: bool) -> list:
        """Get the neighboring parameters at an index."""
        original_value = param_config[index]
        params = list(
            set(
                neighbor[index]
                for neighbor in self.get_neighbors(param_config, neighbor_method)
                if neighbor[index] != original_value
            )
        )
        if randomize:
            shuffle(params)
        return params

    def order_param_configs(
        self, param_configs: List[tuple], order: List[int], randomize_in_params=True
    ) -> List[tuple]:
        """Order a list of parameter configurations based on the indices of the parameters given, starting at 0. If randomize_params is true, the order within parameters is shuffled."""
        if len(order) != self.num_params:
            raise ValueError(
                f"The length of the order ({len(order)}) must be equal to the number of parameters ({self.num_params})"
            )
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
                if (
                    param_config[param_index] != base_comparison[param_index]
                    and param_config not in ordered_param_configs
                ):
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

    def to_ax_searchspace(self):
        """Convert this searchspace to an Ax SearchSpace."""
        from ax import ChoiceParameter, FixedParameter, ParameterType, SearchSpace

        # create searchspace
        ax_searchspace = SearchSpace([])

        # add the parameters
        for param_name, param_values in self.tune_params.items():
            if len(param_values) == 0:
                continue

            # convert the types
            assert all(
                isinstance(param_values[0], type(v)) for v in param_values
            ), f"Parameter values of mixed types are not supported: {param_values}"
            param_type_mapping = {
                str: ParameterType.STRING,
                int: ParameterType.INT,
                float: ParameterType.FLOAT,
                bool: ParameterType.BOOL,
            }
            param_type = param_type_mapping[type(param_values[0])]

            # add the parameter
            if len(param_values) == 1:
                ax_searchspace.add_parameter(FixedParameter(param_name, param_type, param_values[0]))
            else:
                ax_searchspace.add_parameter(ChoiceParameter(param_name, param_type, param_values))

        # add the constraints
        raise NotImplementedError(
            "Conversion to Ax SearchSpace has not been fully implemented as Ax Searchspaces can't capture full complexity."
        )
        # return ax_searchspace
