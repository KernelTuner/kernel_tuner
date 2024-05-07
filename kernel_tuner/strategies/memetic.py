import logging
import ray
import os
import sys
import copy

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.runners.parallel import ParallelRunner
from kernel_tuner.runners.simulation import SimulationRunner
from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner.runners.ray.cache_manager import CacheManager
from kernel_tuner.strategies.common import check_num_devices
from kernel_tuner.util import get_num_devices

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
    options = tuning_options.strategy_options
    simulation_mode = True if isinstance(runner, SimulationRunner) else False
    local_search = options.get('local_search', 'greedy_ils')
    global_search = options.get('global_search', "genetic_algorithm")
    max_feval = options.get("max_fevals", 100)
    alsd = options.get("alsd", 2) # Adaptive Local Search Depth (ALSD)
    lsd = options.get("lsd", 25) # Local Search Depth (LSD)
    maxiter = options.get("maxiter", 2)
    popsize = options.get("popsize", 20)

    if local_search in ls_strategies_list:
        tuning_options["ensemble"] = [local_search] * popsize
    else:
        raise ValueError("Provided local search ensemble are not all local search strategies")

    if global_search in pop_based_strategies_list:
        global_search = strategy_map[global_search]
    else:
        raise ValueError("Provided population based strategy is not a population based strategy")
    
    tuning_options.strategy_options["population"] = searchspace.get_random_sample(popsize)

    # Initialize Ray
    if not ray.is_initialized():
        check_num_devices(popsize, simulation_mode, runner)
        os.environ["RAY_DEDUP_LOGS"] = "0"
        ray.init(include_dashboard=True, ignore_reinit_error=True)
    num_gpus = get_num_devices(runner.kernel_source.lang, simulation_mode=simulation_mode)
    # Create cache manager and actors
    cache_manager = CacheManager.remote(tuning_options)
    if simulation_mode:
        pop_runner = runner
    else:
        pop_runner = ParallelRunner(runner.kernel_source, runner.kernel_options, runner.device_options, 
                                runner.iterations, runner.observers, num_gpus=num_gpus, cache_manager=cache_manager,
                                simulation_mode=simulation_mode)
    
    all_results = []
    all_results_dict = {}
    feval = 0
    while feval < max_feval:
        print(f"DEBUG: --------------------NEW ITERATION--------feval = {feval}------------", file=sys.stderr)
        feval_left = max_feval - feval
        if feval_left < lsd + maxiter * popsize:
            maxiter = feval_left // popsize
            if maxiter == 1: # It doesnt make sense to have one generation for global search, so we give all final resources to local search
                maxiter = 0
                lsd = feval_left
            lsd = feval_left - maxiter * popsize
        print(f"DEBUG: maxiter * popsize = {maxiter * popsize}, lsd = {lsd}", file=sys.stderr)

        # Global Search (GS)
        print(f"DEBUG:=================Global Search=================", file=sys.stderr)
        tuning_options.strategy_options["maxiter"] = maxiter
        pop_start_gs = copy.deepcopy(tuning_options.strategy_options["population"])
        results = global_search.tune(searchspace, pop_runner, tuning_options)
        add_to_results(all_results, all_results_dict, results, tuning_options.tune_params)
        feval += maxiter * popsize

        pop_start_gs_res = get_pop_results(pop_start_gs, all_results_dict)
        pop_end_gs = copy.deepcopy(tuning_options.strategy_options["population"])
        pop_end_gs_res = get_pop_results(pop_end_gs, all_results_dict)
        afi_gs = calculate_afi(pop_start_gs_res, pop_end_gs_res, maxiter, all_results_dict)

        # Local Search (LS)
        print(f"DEBUG:=================Local Search=================", file=sys.stderr)
        tuning_options.strategy_options["max_fevals"] = lsd
        pop_start_ls = copy.deepcopy(tuning_options.strategy_options["candidates"])
        results = ensemble.tune(searchspace, runner, tuning_options, cache_manager=cache_manager)
        add_to_results(all_results, all_results_dict, results, tuning_options.tune_params)
        feval += lsd

        pop_start_ls_res = get_pop_results(pop_start_ls, all_results_dict)
        pop_end_ls = copy.deepcopy(tuning_options.strategy_options["candidates"])
        pop_end_ls_res = get_pop_results(pop_end_ls, all_results_dict)
        afi_ls = calculate_afi(pop_start_ls_res, pop_end_ls_res, lsd, all_results_dict)

        # Adaptive Local Search Depth (ALSD)
        if afi_gs is not None and afi_ls is not None:
            if afi_ls > afi_gs:
                lsd += alsd
            elif afi_ls < afi_gs:
                lsd -= alsd if lsd - alsd > 5 else 5
                print(f"DEBUG: Adaptive Local Search Depth (ALSD) lsd = {lsd}", file=sys.stderr)

    ray.kill(cache_manager)

    return results

def calculate_afi(pop_before_rs, pop_after_rs, feval, results):
    # Average Fitness Increment (AFI)
    delta_fitness = fitness_increment(pop_before_rs, pop_after_rs)
    afi = delta_fitness / feval if feval > 0 else None
    print(f"DEBUG:calculate_afi afi: {afi}", file=sys.stderr)
    return afi

def fitness_increment(pop_before, pop_after):
    if len(pop_before) != len(pop_after):
        raise ValueError("populations must have the same size.")
    
    sum_before = sum(t for t in pop_before if isinstance(t, float))
    sum_after = sum(t for t in pop_after if isinstance(t, float))
    difference_sum = sum_before - sum_after
    print(f"DEBUG:fitness_increment difference_sum: {difference_sum}", file=sys.stderr)
    return difference_sum

def get_pop_results(pop, results):
    print(f"DEBUG:get_pop_results pop = {pop}", file=sys.stderr)
    times = []
    for entry in pop:
        key = ','.join(map(str, entry))
        if key in results:
            time = results[key]
            times.append(time)
        else:
            times.append(None)

    print(f"DEBUG:get_pop_results times = {times}", file=sys.stderr)
    return times

def add_to_results(all_results, all_results_dict, results, tune_params):
    for result in results:
        key = ",".join(str(result[param]) for param in tune_params)
        all_results_dict[key] = result["time"]
        all_results.append(result)