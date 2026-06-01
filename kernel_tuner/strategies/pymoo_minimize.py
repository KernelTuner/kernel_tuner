"""The Pymoo strategy that uses a minimizer method for searching through the parameter space."""

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.termination import NoTermination, Termination
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.pntx import SinglePointCrossover, TwoPointCrossover
from pymoo.util.ref_dirs import get_reference_directions

from kernel_tuner import util
from kernel_tuner.runners.runner import Runner
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
    get_strategy_docstring,
    get_options
)

_SUPPORTED_ALGOS = {
    "nsga2": NSGA2,
    "nsga3": NSGA3
}

crossover_oper_dict = {
    "uniform-crossover": UniformCrossover,
    "single-point-crossover": SinglePointCrossover,
    "two-point-crossover": TwoPointCrossover,
}
supported_crossover_oper_names = list(crossover_oper_dict.keys())

_options = {
    "pop_size": ("Initial population size", 20),
    "crossover_operator": (f"The crossover operator, can be one of {supported_crossover_oper_names}", "two-point-crossover"),
    "crossover_prob": ("Crossover probability", 1.0),
    "mutation_prob": ("Mutation probability", 0.1),
    "ref_dirs_list": ("The list of reference directions on the unit hyperplane in the objective space to guide NSGA-III, see https://pymoo.org/misc/reference_directions.html for more information.", []),
}

_option_defaults = { key: option_pair[1] for key, option_pair in _options.items() }


def tune(
    searchspace: Searchspace,
    runner: Runner,
    tuning_options,
):
    algo_name: str = tuning_options.strategy
    strategy_options = tuning_options.strategy_options

    algo_name = algo_name.lower()
    if algo_name in _SUPPORTED_ALGOS:
        algorithm = _SUPPORTED_ALGOS[algo_name]
    else:
        raise ValueError(f"\"{algo_name}\" is not supported. The supported algorithms are: {_SUPPORTED_ALGOS.keys}\n")

    pop_size, crossover_oper, crossover_prob, mutation_prob, ref_dirs_list = get_options(strategy_options, _options, unsupported=["x0"])

    if algo_name == "nsga3" and len(ref_dirs_list) == 0:
        ref_dirs_list = get_reference_directions("energy", len(tuning_options.objective), pop_size)

    if crossover_oper in crossover_oper_dict:
        crossover_oper = crossover_oper_dict[crossover_oper]
    else:
        raise ValueError(f"Unsupported crossover method {crossover_oper}")

    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=False)

    problem = TuningProblem(
        cost_func = cost_func,
        n_var = len(tuning_options.tune_params),
        n_obj = len(tuning_options.objective),
    )

    sampling = TuningSearchspaceRandomSampling(searchspace)
    crossover = crossover_oper(prob = crossover_prob)
    mutation = TuningParamConfigNeighborhoodMutation(prob = mutation_prob, searchspace = searchspace)
    repair = TuningParamConfigRepair()
    eliminate_duplicates = TuningParamConfigDuplicateElimination()

    algo = algorithm(pop_size = pop_size,
                ref_dirs = ref_dirs_list if algo_name == "nsga3" else None,
                sampling = sampling,
                crossover = crossover,
                mutation = mutation,
                repair = repair,
                eliminate_duplicates = eliminate_duplicates,
            )

    # CostFunc throws exception when done, so isn't really needed
    termination = None
    if "max_fevals" in tuning_options.strategy_options or "time_limit" in tuning_options.strategy_options:
        termination = NoTermination()

    try:
        algo.setup(
            problem,
            termination = termination,
            verbose = tuning_options.verbose,
            progress = tuning_options.verbose,
            seed = tuning_options.seed,
        )

        while algo.has_next():
            algo.next()

    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(f"Stopped because of {e}")

    return cost_func.results


tune.__doc__ = get_strategy_docstring("Pymoo minimize", _options)


class TuningProblem(ElementwiseProblem):
    """ Class used by PyMoo to wrap the objective function and calculate Pareto front. """
    def __init__(
        self,
        cost_func: CostFunc,
        n_var: int,
        n_obj: int,
        **kwargs,
    ):
        super().__init__(
            n_var = n_var,
            n_obj = n_obj,
            **kwargs,
        )
        self.cost_func = cost_func
        self.searchspace = cost_func.searchspace
        self.tuning_options = cost_func.tuning_options

    def _evaluate( self, x, out, *args, **kwargs, ):
        # A copy of `x` is made to make sure sharing does not happen.
        F = self.cost_func(tuple(x))
        out["F"] = F

    def _calc_pareto_front( self, *args, **kwargs, ):
        # Can only compute the pareto front if we are in simulation mode.
        if not self.tuning_options.simulation_mode:
            return None

        objectives = self.tuning_options.objective
        higher_is_better = self.tuning_options.objective_higher_is_better
        pareto_results = util.get_pareto_results(
            list(self.tuning_options.cache.values()),
            objectives,
            higher_is_better,
        )

        pareto_front_list = []
        for res in pareto_results:
            cost = util.get_result_cost(res, objectives, higher_is_better)
            pareto_front_list.append(cost)

        return np.array(pareto_front_list, dtype=float)


class TuningTermination(Termination):
    """ Class used by PyMoo to detect termination. """
    def __init__( self, tuning_options, ):
        super().__init__()
        self.tuning_options = tuning_options
        self.reason = None

    def _update(
        self,
        algorithm,
    ):
        try:
            self.tuning_options.budget.raise_exception_if_done()
            print(f"progress: {self.tuning_options.budget.get_fraction_consumed()}")
            return 0.0
        except util.StopCriterionReached as e:
            self.terminate()
            self.reason = e
            return 1.0


class TuningSearchspaceRandomSampling(Sampling):
    """ Class used by PyMoo to generate a random sample config. """
    def __init__( self, searchspace, ):
        super().__init__()
        self.searchspace = searchspace

    def _do( self, problem, n_samples: int, **kwargs, ):
        sample = self.searchspace.get_random_sample(n_samples)
        return np.array(sample, dtype=object)


class TuningParamConfigNeighborhoodMutation(Mutation):
    """ Class used by PyMoo to mutate configs. """
    def __init__(
        self,
        prob,
        searchspace: Searchspace,
        **kwargs
    ):
        super().__init__(
            prob = prob,
            **kwargs,
        )
        self.searchspace = searchspace

    def _do(
        self,
        problem: TuningProblem,
        X: np.ndarray,
        **kwargs,
    ):
        for X_index in range(X.shape[0]):
            params_config_tuple = tuple(X[X_index])
            neighbors_indices = self.searchspace.get_neighbors_indices_no_cache(params_config_tuple, neighbor_method="Hamming")
            if len(neighbors_indices) > 0:
                neighbor_index = neighbors_indices[np.random.choice(len(neighbors_indices))]
                neighbor = self.searchspace.get_param_configs_at_indices([neighbor_index])[0]
                X[X_index] = np.array(neighbor, dtype=object)

        return X


class TuningParamConfigRepair(Repair):
    """ Class used by PyMoo to repair invalid configs. """
    def _do(
        self,
        problem: TuningProblem,
        X: np.ndarray,
        **kwargs,
    ):
        for X_index in range(X.shape[0]):
            params_config_tuple = tuple(X[X_index])
            if problem.searchspace.is_param_config_valid(params_config_tuple):
                continue
            for neighbor_method in ["strictly-adjacent", "adjacent", "Hamming"]:
                neighbors_indices = problem.searchspace.get_neighbors_indices_no_cache(params_config_tuple, neighbor_method)
                if len(neighbors_indices) > 0:
                    neighbor_index = neighbors_indices[np.random.choice(len(neighbors_indices))]
                    neighbor = problem.searchspace.get_param_configs_at_indices([neighbor_index])[0]
                    X[X_index] = np.array(neighbor, dtype=object)
                    break

        return X


class TuningParamConfigDuplicateElimination(ElementwiseDuplicateElimination):
    """ Class needed by PyMoo to eliminate duplicates. """
    def is_equal(self, a, b):
        return np.all(a.X == b.X)
