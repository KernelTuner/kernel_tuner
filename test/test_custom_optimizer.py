
### The following was generating using the LLaMEA prompt and OpenAI o1

import numpy as np

from kernel_tuner.strategies.wrapper import OptAlg

class HybridDELocalRefinement(OptAlg):
    """
    A two-phase differential evolution with local refinement, intended for BBOB-type
    black box optimization problems in [-5,5]^dim.

    One-line idea: A two-phase hybrid DE with local refinement that balances global
    exploration and local exploitation under a strict function evaluation budget.
    """

    def __init__(self):
        super().__init__()
        self.costfunc_kwargs = {"scaling": True, "snap": True}
        # You can adjust these hyperparameters based on experimentation/tuning:
        self.F = 0.8        # Differential weight
        self.CR = 0.9       # Crossover probability
        self.local_search_freq = 10  # Local refinement frequency in generations

    def __call__(self, func, searchspace):
        """
        Optimize the black box function `func` in [-5,5]^dim, using
        at most self.budget function evaluations.

        Returns:
            best_params: np.ndarray representing the best parameters found
            best_value: float representing the best objective value found
        """
        self.dim = searchspace.num_params
        self.population_size = round(min(min(50, 10 * self.dim), np.ceil(searchspace.size / 3)))  # Caps for extremely large dim

        # 1. Initialize population
        lower_bound, upper_bound = -5.0, 5.0
        pop = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))

        # Evaluate initial population
        evaluations = 0
        fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            fitness[i] = func(pop[i])
            evaluations += 1

        # Track best solution
        best_idx = np.argmin(fitness)
        best_params = pop[best_idx].copy()
        best_value = fitness[best_idx]

        # 2. Main evolutionary loop
        gen = 0
        while func.budget_spent_fraction < 1.0 and evaluations < searchspace.size:
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
                if func.budget_spent_fraction > 1.0:
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
            if gen % self.local_search_freq == 0 and func.budget_spent_fraction < 1.0:
                best_params, best_value, evaluations = self._local_refinement(
                    func, best_params, best_value, evaluations, lower_bound, upper_bound
                )

        return best_params, best_value

    def _local_refinement(self, func, best_params, best_value, evaluations, lb, ub):
        """
        Local refinement around the best solution found so far.
        Uses a quick 'perturb-and-accept' approach in a shrinking neighborhood.
        """
        # Neighborhood size shrinks as the budget is consumed
        step_size = 0.2 * (1.0 - func.budget_spent_fraction)

        for _ in range(5):  # 5 refinements each time
            if func.budget_spent_fraction >= 1.0:
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
from kernel_tuner import tune_kernel, tune_kernel_T1
from kernel_tuner.strategies.wrapper import OptAlgWrapper
from pathlib import Path

from .test_runners import env   # noqa: F401

cache_filename = Path(__file__).parent.resolve() / "test_cache_file.json"

def test_OptAlgWrapper(env):
    kernel_name, kernel_string, size, args, tune_params = env

    # Instantiate LLaMAE optimization algorithm
    optimizer = HybridDELocalRefinement()

    # Wrap the algorithm class in the OptAlgWrapper
    # for use in Kernel Tuner
    strategy = OptAlgWrapper(optimizer)
    strategy_options = { 'max_fevals': 15 }

    # Call the tuner
    res, _ = tune_kernel(kernel_name, kernel_string, size, args, tune_params,
                strategy=strategy, strategy_options=strategy_options, cache=cache_filename,
                simulation_mode=True, verbose=True)
    assert len(res) == strategy_options['max_fevals']

def test_OptAlgWrapper_T1(env):
    kernel_name, kernel_string, size, args, tune_params = env

    strategy = "HybridDELocalRefinement"
    strategy_options = {
        "max_fevals": 15,
        "custom_search_method_path": Path(__file__).resolve(),
        "constraint_aware": False,
    }
    iterations = 1
    
    res, _ = tune_kernel_T1(
        Path(__file__).parent.resolve() / "test_cache_file_T1_input.json",
        cache_filename,
        device="NVIDIA RTX A4000",
        objective="time",
        objective_higher_is_better=False,
        simulation_mode=True,
        output_T4=False,
        iterations=iterations,
        strategy=strategy,
        strategy_options=strategy_options,
    )

    assert len(res) == strategy_options['max_fevals']
