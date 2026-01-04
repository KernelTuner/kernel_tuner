"""The Pymoo strategy that uses a minimizer method for searching through the parameter space."""

from typing import assert_never
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.termination import NoTermination, Termination
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.operators.crossover.pntx import TwoPointCrossover

from kernel_tuner import util
from kernel_tuner.runners.runner import Runner
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
    get_strategy_docstring,
)

from enum import StrEnum

class SupportedAlgos(StrEnum):
    NSGA2 = "nsga2"
    NSGA3 = "nsga3"

supported_algos = [ algo.value for algo in SupportedAlgos ]

supported_crossover_opers = [
    # "uniform-crossover",
    # "single-point-crossover",
    "two-point-crossover",
]

_options = {
    "pop_size": ("Initial population size", 20),
    "crossover_operator": ("The crossover operator", "two-point-crossover"),
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
    if algo_name not in SupportedAlgos:
        raise ValueError(f"\"{algo_name}\" is not supported. The supported algorithms are: {supported_algos}\n")
    else:
        algo_name = SupportedAlgos(algo_name)

    pop_size = strategy_options.get("pop_size", _option_defaults["pop_size"])
    crossover_prob = strategy_options.get("crossover_prob", _option_defaults["crossover_prob"])
    mutation_prob = strategy_options.get("mutation_prob", _option_defaults["mutation_prob"])
    ref_dirs_list = strategy_options.get("ref_dirs_list", _option_defaults["ref_dirs_list"])

    if algo_name == "nsga3" and len(ref_dirs_list) == 0:
        raise ValueError("NSGA-III requires reference directions to be specified, but they are missing.")

    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=False)

    problem = TuningProblem(
        cost_func = cost_func,
        n_var = len(tuning_options.tune_params),
        n_obj = len(tuning_options.objective),
    )

    sampling = TuningSearchspaceRandomSampling(searchspace)
    crossover = TwoPointCrossover(prob = crossover_prob)
    mutation = TuningParamConfigNeighborhoodMutation(prob = mutation_prob, searchspace = searchspace)
    repair = TuningParamConfigRepair()
    eliminate_duplicates = TuningParamConfigDuplicateElimination()

    # algorithm_type = get_algorithm(method)
    algo: Algorithm
    match algo_name:
        case SupportedAlgos.NSGA2:
            algo = NSGA2(
                pop_size = pop_size,
                sampling = sampling,
                crossover = crossover,
                mutation = mutation,
                repair = repair,
                eliminate_duplicates = eliminate_duplicates,
            )
        case SupportedAlgos.NSGA3:
            algo = NSGA3(
                pop_size = pop_size,
                ref_dirs = ref_dirs_list,
                sampling = sampling,
                crossover = crossover,
                mutation = mutation,
                repair = repair,
                eliminate_duplicates = eliminate_duplicates,
            )
        case _ as unreachable:
            assert_never(unreachable)

    # TODO:
    # - CostFunc throws exception when done, so isn't really needed
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

    results = cost_func.results

    if results and tuning_options.verbose:
        print(f"{results.message=}")

    return results


tune.__doc__ = get_strategy_docstring("Pymoo minimize", _options)


class TuningProblem(ElementwiseProblem):
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
        # A copy of `x` is made to make sure sharing does not happen
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

        pareto_front_list = list()
        for res in pareto_results:
            cost = util.get_result_cost(res, objectives, higher_is_better)
            pareto_front_list.append(cost)

        return np.array(pareto_front_list, dtype=float)


class TuningTermination(Termination):
    def __init__( self, tuning_options, ):
        super().__init__()
        self.tuning_options = tuning_options
        self.reason = None

    def _update(
        self,
        algorithm,
    ):
        try:
            util.check_stop_criterion(self.tuning_options)
            print(f"progress: {len(self.tuning_options.unique_results) / self.tuning_options.max_fevals}")
            return 0.0
        except util.StopCriterionReached as e:
            self.terminate()
            self.reason = e
            return 1.0


class TuningSearchspaceRandomSampling(Sampling):
    def __init__( self, searchspace, ):
        super().__init__()
        self.searchspace = searchspace

    def _do( self, problem, n_samples: int, **kwargs, ):
        sample = self.searchspace.get_random_sample(n_samples)
        return np.array(sample, dtype=object)


class TuningParamConfigNeighborhoodMutation(Mutation):
    def __init__(
        self,
        prob,
        searchspace: Searchspace,
        **kwargs
    ):
        super().__init__(
            prob = prob,
            # prob_var = None,
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

    def is_equal(self, a, b):
        return np.all(a.X == b.X)
