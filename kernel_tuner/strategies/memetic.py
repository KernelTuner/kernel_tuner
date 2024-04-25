import logging
import ray
import os

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.runners.parallel import ParallelRunner
from kernel_tuner.runners.simulation import SimulationRunner
from kernel_tuner.runners.ray.cache_manager import CacheManager
from kernel_tuner.strategies.common import setup_resources

from kernel_tuner.strategies import (
    basinhopping,
    bayes_opt,
    brute_force,
    diff_evo,
    dual_annealing,
    firefly_algorithm,
    genetic_algorithm,
    greedy_ils,
    greedy_mls,
    minimize,
    mls,
    ordered_greedy_mls,
    pso,
    random_sample,
    simulated_annealing,
    ensemble,
    memetic
)

strategy_map = {
    "brute_force": brute_force,
    "random_sample": random_sample,
    "minimize": minimize,
    "basinhopping": basinhopping,
    "diff_evo": diff_evo,
    "genetic_algorithm": genetic_algorithm,
    "greedy_mls": greedy_mls,
    "ordered_greedy_mls": ordered_greedy_mls,
    "greedy_ils": greedy_ils,
    "dual_annealing": dual_annealing,
    "mls": mls,
    "pso": pso,
    "simulated_annealing": simulated_annealing,
    "firefly_algorithm": firefly_algorithm,
    "bayes_opt": bayes_opt,
}

# Pseudo code from "Memetic algorithms and memetic computing optimization: A literature review" by Ferrante Neri and Carlos Cotta
# function BasicMA (in P: Problem, in par: Parameters):
# Solution; 
# begin 
#     pop ← Initialize(par, P); 
#     repeat 
#         newpop1 ← Cooperate(pop, par, P); 
#         newpop2 ← Improve(newpop1, par, P); 
#         pop ← Compete (pop, newpop2); 
#         if Converged(pop) then 
#             pop ← Restart(pop, par); 
#         end 
#     until TerminationCriterion(par); 
#     return GetNthBest(pop, 1); 
# end

ls_strategies_list = {
    "greedy_mls",
    "ordered_greedy_mls",
    "greedy_ils",
    "mls",
    "hill_climbing"
}

pop_based_strategies_list = {
    "genetic_algorithm",
    "differential_evolution",
    "pso"
}


def tune(searchspace: Searchspace, runner, tuning_options):
    simulation_mode = True if isinstance(runner, SimulationRunner) else False
    ls_strategies = ["greedy_ils", "greedy_ils", "greedy_ils", "greedy_ils"]
    pop_based_strategy = "genetic_algorithm"
    iterations = 10

    if set(ls_strategies) <= ls_strategies_list:
        tuning_options["ensemble"] = ls_strategies
    else:
        raise ValueError("Provided local search ensemble are not all local search strategies")

    if pop_based_strategy in pop_based_strategies_list:
        pop_based_strategy = strategy_map[pop_based_strategy]
    else:
        raise ValueError("Provided population based strategy is not a population based strategy")
    
    tuning_options.strategy_options["candidates"] = searchspace.get_random_sample(len(ls_strategies))
    tuning_options.strategy_options["max_fevals"] = 10
    tuning_options.strategy_options["maxiter"] = 10

    resources = setup_resources(len(ls_strategies), simulation_mode, runner)
    # Initialize Ray
    if not ray.is_initialized():
        os.environ["RAY_DEDUP_LOGS"] = "0"
        ray.init(resources=resources, include_dashboard=True, ignore_reinit_error=True)
    # Create cache manager and actors
    cache_manager = CacheManager.options(resources={"cache_manager_cpu": 1}).remote(tuning_options)
    pop_runner = ParallelRunner(runner.kernel_source, runner.kernel_options, runner.device_options, 
                                runner.iterations, runner.observers, cache_manager=cache_manager,
                                resources=resources)
    
    for i in range(iterations):
        print(f"Memetic algorithm iteration {i}")
        print(f"start local search ensemble with candidates = {tuning_options.strategy_options['candidates']}")
        ensemble.tune(searchspace, runner, tuning_options, cache_manager=cache_manager)
        print(f"start pop base algo with population = {tuning_options.strategy_options['population']}")
        results = pop_based_strategy.tune(searchspace, pop_runner, tuning_options)

    return results