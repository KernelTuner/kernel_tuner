"""The Pymoo strategy that uses a minimizer method for searching through the parameter space."""

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.termination import NoTermination, Termination
from pymoo.core.repair import Repair
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.igd import IGD

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
    get_strategy_docstring,
    setup_method_arguments,
)

# TODO: Add the PyMOO algorithms
supported_methods = [
    "NSGA2",
    "NSGA3",
]

_options = {
    "method": (f"Pymoo optimization algorithm to use, choose any from {supported_methods}", "NSGA2"),
    "pop_size": ("Initial population size", 100),
}


def tune(
    searchspace: Searchspace,
    runner,
    tuning_options,
):
    strategy_options = tuning_options.strategy_options

    if "method" in strategy_options:
        method = strategy_options["method"]
    else:
        (_, method) = _options["method"]
    print(f"{method=}")

    if "pop_size" in strategy_options:
        pop_size = strategy_options["pop_size"]
    else:
        (_, pop_size) = _options["pop_size"]
    print(f"{pop_size=}")

    # scale variables in x to make 'eps' relevant for multiple variables
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=False)

    bounds, x0, _ = cost_func.get_bounds_x0_eps()
    kwargs = setup_method_arguments(method, bounds)

    problem = TuningProblem(
        cost_func=cost_func,
        n_var=len(tuning_options.tune_params),
        n_obj=len(tuning_options.objective),
    )

    # algorithm_type = get_algorithm(method)
    algorithm = None
    if method == "NSGA2":
        algorithm = NSGA2(
            pop_size = pop_size,
            sampling = SearchspaceRandomSampling(searchspace),
            crossover = TwoPointCrossover(),
            mutation = MutateToNeighbor(searchspace, prob = 0.5),
            repair = RepairConfig(),
            # save_history = True,
        )
    elif method == "NSGA3":
        algorithm = NSGA3(
            pop_size = pop_size,
            ref_dirs = get_reference_directions("das-dennis", len(tuning_options.objective), n_partitions = 26),
            sampling = SearchspaceRandomSampling(searchspace),
            crossover = UniformCrossover(prob = 0.6),
            mutation = MutateToNeighbor(searchspace, prob = 0.5),
            # repair = MyRepair(),
            # save_history = True,
        )

    # TODO:
    # - CostFunc throws exception when done, so isn't really needed
    termination = None
    if "max_fevals" in tuning_options.strategy_options or "time_limit" in tuning_options.strategy_options:
        termination = NoTermination()

    pf = problem.pareto_front()
    igd_ind = IGD(pf, zero_to_one=True)

    try:
        _ = algorithm.setup(
            problem,
            # termination = termination,
            termination=("n_gen", 20),
            seed=1,
            verbose=True,
        )

        while algorithm.has_next():
            algorithm.next()

            illegal_count = cost_func.illegal_config_count
            total_count = cost_func.total_config_count
            print(f"config valid: {total_count - illegal_count}/{total_count} ({100 * (1 - (illegal_count / total_count)):.4}%)")

            print("IGD: ", igd_ind(algorithm.opt.get("F")))

    except util.StopCriterionReached as e:
        if tuning_options.verbose:
            print(f"Stopped because of {e}")

    opt_result = cost_func.results

    if opt_result and tuning_options.verbose:
        print(f"{opt_result.message=}")

    return opt_result


tune.__doc__ = get_strategy_docstring("Pymoo minimize", _options)


class TuningProblem(ElementwiseProblem):
    def __init__(
        self,
        cost_func: CostFunc,
        n_var,
        n_obj,
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

    def _evaluate(
        self,
        x,
        out,
        *args,
        **kwargs,
    ):
        F = self.cost_func(x)
        out["F"] = F

    def _calc_pareto_front(
        self,
        *args,
        **kwargs
    ) -> np.ndarray | None:
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

        return np.array(pareto_front_list)


class TuningTermination(Termination):
    def __init__(
        self,
        tuning_options,
    ):
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


class SearchspaceRandomSampling(Sampling):
    def __init__(
        self,
        searchspace,
    ):
        super().__init__()
        self.searchspace = searchspace

    def _do(
        self,
        problem,
        n_samples: int,
        **kwargs,
    ):
        X = self.searchspace.get_random_sample(n_samples)
        return X


class MutateToNeighbor(Mutation):
    def __init__(
        self,
        searchspace: Searchspace,
        prob=1.0,
        prob_var=None,
        **kwargs
    ):
        super().__init__(
            prob=prob,
            prob_var=prob_var,
            **kwargs,
        )
        self.searchspace = searchspace

    def _do(
        self,
        problem: TuningProblem,
        X: np.ndarray,
        **kwargs,
    ):
        for ind_index in range(X.shape[0]):
            params_config_tuple = tuple(X[ind_index])
            neighbors_indices = self.searchspace.get_neighbors_indices_no_cache(params_config_tuple, neighbor_method="Hamming")
            if len(neighbors_indices) > 0:
                neighbor_index = neighbors_indices[np.random.choice(len(neighbors_indices))]
                neighbor = self.searchspace.get_param_configs_at_indices([neighbor_index])[0]
                X[ind_index] = np.array(neighbor)

        return X


class RepairConfig(Repair):

    def _do(
        self,
        problem: TuningProblem,
        X : np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        for ind_index in range(X.shape[0]):
            params_config_tuple = tuple(X[ind_index])
            if problem.searchspace.is_param_config_valid(params_config_tuple):
               continue
            for neighbor_method in ["strictly-adjacent", "adjacent", "Hamming"]:
                neighbors_indices = problem.searchspace.get_neighbors_indices_no_cache(params_config_tuple, neighbor_method)
                if len(neighbors_indices) > 0:
                    neighbor_index = neighbors_indices[np.random.choice(len(neighbors_indices))]
                    neighbor = problem.searchspace.get_param_configs_at_indices([neighbor_index])[0]
                    X[ind_index] = np.array(neighbor)
                    break

        return X
