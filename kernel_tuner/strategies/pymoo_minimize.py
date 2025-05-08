"""The Pymoo strategy that uses a minimizer method for searching through the parameter space."""

import numpy as np

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import (
    CostFunc,
    get_options,
    get_strategy_docstring,
    setup_method_arguments,
    setup_method_options,
)
from kernel_tuner.strategies.genetic_algorithm import mutate

from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.ux import UX
from pymoo.algorithms.moo.nsga2 import NSGA2

# TODO: Add the PyMOO algorithms
supported_methods = []

_options = dict(method=(f"Pymoo optimization algorithm to use, choose any from {supported_methods}", ""))

def tune(searchspace: Searchspace, runner, tuning_options):

    # TODO:
    # The idea is to create a Problem, Algorithm, and Termination
    # then use to run `pymoo.optimize.minimize`
    # so I basically need to write some adapter/integration code

    method = get_options(tuning_options.strategy_options, _options)[0]

    # scale variables in x to make 'eps' relevant for multiple variables
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=False)

    bounds, x0, _ = cost_func.get_bounds_x0_eps()
    kwargs = setup_method_arguments(method, bounds)
    options = setup_method_options(method, tuning_options)

    problem = KernelTunerProblem(
        f = cost_func,
        n_var = len(tuning_options.tune_params),
        n_obj = len(tuning_options.objective),
    )

    # TODO: make a pymoo.core.algorithm.Algorithm
    # * use `method` to select the algorithm
    # * use etc to define the algorithm
    
    # algorithm_type = get_algorithm
    algorithm = NSGA2(
        pop_size=100,
        sampling=SearchspaceRandomSampling(searchspace),
        crossover=UX(prob=0.6),
        mutation=MutateToNeighbor(searchspace, prob=0.5),
    )

    # TODO:
    # - CostFunc throws exception when done, so isn't really needed
    termination = None # pymoo.core.termination.Termination()

    opt_result = None
    try:
        opt_result = minimize(problem, algorithm, termination)
    except util.StopCriterionReached as e:
        print(f"Stopped because of {e}")
        if tuning_options.verbose:
            print(e)

    if opt_result and tuning_options.verbose:
        print(f"{opt_result.message=}")

    # print(f"{opt_result.message=}")
    # print(f"{cost_func.results=}")
    return cost_func.results


tune.__doc__ = get_strategy_docstring("Pymoo minimize", _options)


class KernelTunerProblem(ElementwiseProblem):
    def __init__(self, f, n_var, n_obj):
        super().__init__(
            n_var = n_var,
            n_obj = n_obj,
        )
        self.f = f

    def _evaluate(self, x, out, *args, **kwargs):
        F = self.f(x)
        out["F"] = F


class SearchspaceRandomSampling(Sampling):
    def __init__(self, searchspace):
        super().__init__()
        self.ss = searchspace

    def _do(self, problem, n_samples, **kwargs):
        X = self.ss.get_random_sample(n_samples)
        return X


class MutateToNeighbor(Mutation):
    def __init__(
            self,
            searchspace : Searchspace,
            prob=1.0,
            prob_var=None,
            **kwargs
        ):
        super().__init__(
            prob=prob,
            prob_var=prob_var,
            **kwargs,
        )
        self.ss = searchspace

    def _do(self, problem, X, **kwargs):
        Xm = np.empty_like(X)
        for i in range(X.shape[0]):
            neighbors = self.ss.get_neighbors_indices_no_cache(tuple(X[i]), neighbor_method="Hamming")
            # copy X[i] to result in case there are no neighbors
            if len(neighbors) > 0:
                Xm[i] = neighbors[np.random.choice(len(neighbors))]
            else:
                Xm[i] = X[i]
        return Xm
