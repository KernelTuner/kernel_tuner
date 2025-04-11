
### The following was generating using the LLaMEA prompt and OpenAI o1

import numpy as np

class HybridDELocalRefinement:
    """
    A two-phase differential evolution with local refinement, intended for BBOB-type
    black box optimization problems in [-5,5]^dim.

    One-line idea: A two-phase hybrid DE with local refinement that balances global
    exploration and local exploitation under a strict function evaluation budget.
    """

    def __init__(self, budget, dim):
        """
        Initialize the optimizer with:
        - budget: total number of function evaluations allowed.
        - dim: dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        # You can adjust these hyperparameters based on experimentation/tuning:
        self.population_size = min(50, 10 * dim)  # Caps for extremely large dim
        self.F = 0.8        # Differential weight
        self.CR = 0.9       # Crossover probability
        self.local_search_freq = 10  # Local refinement frequency in generations

    def __call__(self, func):
        """
        Optimize the black box function `func` in [-5,5]^dim, using
        at most self.budget function evaluations.

        Returns:
            best_params: np.ndarray representing the best parameters found
            best_value: float representing the best objective value found
        """
        # Check if we have a non-positive budget
        if self.budget <= 0:
            raise ValueError("Budget must be a positive integer.")

        # 1. Initialize population
        lower_bound, upper_bound = -5.0, 5.0
        pop = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))

        # Evaluate initial population
        evaluations = 0
        fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            fitness[i] = func(pop[i])
            evaluations += 1
            if evaluations >= self.budget:
                break

        # Track best solution
        best_idx = np.argmin(fitness)
        best_params = pop[best_idx].copy()
        best_value = fitness[best_idx]

        # 2. Main evolutionary loop
        gen = 0
        while evaluations < self.budget:
            gen += 1
            for i in range(self.population_size):
                # DE mutation: pick three distinct indices
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[idxs]
                mutant = a + self.F * (b - c)

                # Crossover
                trial = np.copy(pop[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial[crossover_points] = mutant[crossover_points]

                # Enforce bounds
                trial = np.clip(trial, lower_bound, upper_bound)

                # Evaluate trial
                trial_fitness = func(trial)
                evaluations += 1
                if evaluations >= self.budget:
                    # If out of budget, wrap up
                    if trial_fitness < fitness[i]:
                        pop[i] = trial
                        fitness[i] = trial_fitness
                        # Update global best
                        if trial_fitness < best_value:
                            best_value = trial_fitness
                            best_params = trial.copy()
                    break

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    # Update global best
                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_params = trial.copy()

            # Periodically refine best solution with a small local neighborhood search
            if gen % self.local_search_freq == 0 and evaluations < self.budget:
                best_params, best_value, evaluations = self._local_refinement(
                    func, best_params, best_value, evaluations, lower_bound, upper_bound
                )

            if evaluations >= self.budget:
                break

        return best_params, best_value

    def _local_refinement(self, func, best_params, best_value, evaluations, lb, ub):
        """
        Local refinement around the best solution found so far.
        Uses a quick 'perturb-and-accept' approach in a shrinking neighborhood.
        """
        # Neighborhood size shrinks as the budget is consumed
        frac_budget_used = evaluations / self.budget
        step_size = 0.2 * (1.0 - frac_budget_used)

        for _ in range(5):  # 5 refinements each time
            if evaluations >= self.budget:
                break
            candidate = best_params + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, lb, ub)
            cand_value = func(candidate)
            evaluations += 1
            if cand_value < best_value:
                best_value = cand_value
                best_params = candidate.copy()

        return best_params, best_value, evaluations




### Testing the Optimization Algorithm Wrapper in Kernel Tuner
import os
from kernel_tuner import tune_kernel
from kernel_tuner.strategies.wrapper import OptAlgWrapper
cache_filename = os.path.dirname(

    os.path.realpath(__file__)) + "/test_cache_file.json"

from .test_runners import env


def test_OptAlgWrapper(env):
    kernel_name, kernel_string, size, args, tune_params = env

    # Instantiate LLaMAE optimization algorithm
    budget = int(15)
    dim = len(tune_params)
    optimizer = HybridDELocalRefinement(budget, dim)

    # Wrap the algorithm class in the OptAlgWrapper
    # for use in Kernel Tuner
    strategy = OptAlgWrapper(optimizer)

    # Call the tuner
    tune_kernel(kernel_name, kernel_string, size, args, tune_params,
                strategy=strategy, cache=cache_filename,
                simulation_mode=True, verbose=True)
