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
from kernel_tuner.strategies.common import check_num_devices, create_actor_on_device, initialize_ray
from kernel_tuner.util import get_num_devices, check_stop_criterion, StopCriterionReached
from kernel_tuner.runners.ray.remote_actor import RemoteActor

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
    alsd = options.get("alsd", 2) # Adaptive Local Search Depth (ALSD)
    lsd = options.get("lsd", 25) # Local Search Depth (LSD)
    maxiter = options.get("maxiter", 2)
    popsize = options.get("popsize", 20)
    max_feval = options.get("max_fevals", None if 'time_limit' in options else 2000)
    print(f"DEBUG: local_search={local_search} global_search={global_search} alsd={alsd} lsd={lsd} maxiter={maxiter} popsize={popsize} max_feval={max_feval}", file=sys.stderr)

    if local_search in ls_strategies_list:
        options["ensemble"] = [local_search] * popsize
    else:
        raise ValueError("Provided local search ensemble are not all local search strategies")

    if global_search in pop_based_strategies_list:
        global_search = strategy_map[global_search]
    else:
        raise ValueError("Provided population based strategy is not a population based strategy")
    
    options["population"] = searchspace.get_random_sample(popsize)

    num_gpus = get_num_devices(runner.kernel_source.lang, simulation_mode=simulation_mode)
    check_num_devices(num_gpus, simulation_mode, runner)
    initialize_ray()
    # Create cache manager, actors and parallel runner
    cache_manager = CacheManager.remote(tuning_options.cache, tuning_options.cachefile)
    num_actors = num_gpus if num_gpus < popsize else popsize
    runner_attributes = [runner.kernel_source, runner.kernel_options, runner.device_options, runner.iterations, runner.observers]
    actors = [create_actor_on_device(*runner_attributes, id=id, cache_manager=cache_manager, simulation_mode=simulation_mode) for id in range(num_actors)]
    pop_runner = ParallelRunner(runner.kernel_source, runner.kernel_options, runner.device_options, 
                                runner.iterations, runner.observers, num_gpus=num_gpus, cache_manager=cache_manager,
                                simulation_mode=simulation_mode, actors=actors)
    
    all_results = []
    all_results_dict = {}
    feval = 0
    afi_gs, afi_ls = None, None
    while (max_feval is None) or feval < max_feval:
        print(f"DEBUG: --------------------NEW ITERATION--------feval = {feval}------------", file=sys.stderr)
        if max_feval is not None:
            maxiter, lsd = distribute_feval(feval, max_feval, maxiter, lsd, popsize, afi_gs, afi_ls)
        print(f"DEBUG: maxiter * popsize = {maxiter * popsize}, lsd = {lsd}", file=sys.stderr)

        # Global Search (GS)
        print(f"DEBUG:=================Global Search=================", file=sys.stderr)
        tuning_options.strategy_options["maxiter"] = maxiter
        pop_start_gs = copy.deepcopy(tuning_options.strategy_options["population"])
        results = global_search.tune(searchspace, pop_runner, tuning_options)
        add_to_results(all_results, all_results_dict, results, tuning_options.tune_params)
        feval += maxiter * popsize
        try:
            check_stop_criterion(tuning_options)
        except StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)
            break

        pop_start_gs_res = get_pop_results(pop_start_gs, all_results_dict)
        pop_end_gs = copy.deepcopy(tuning_options.strategy_options["population"])
        pop_end_gs_res = get_pop_results(pop_end_gs, all_results_dict)
        afi_gs = calculate_afi(pop_start_gs_res, pop_end_gs_res, maxiter, all_results_dict)

        # Local Search (LS)
        print(f"DEBUG:=================Local Search=================", file=sys.stderr)
        tuning_options.strategy_options["max_fevals"] = lsd * popsize
        pop_start_ls = copy.deepcopy(tuning_options.strategy_options["candidates"])
        results = ensemble.tune(searchspace, runner, tuning_options, cache_manager=cache_manager, actors=actors)
        add_to_results(all_results, all_results_dict, results, tuning_options.tune_params)
        feval += lsd * popsize
        try:
            check_stop_criterion(tuning_options)
        except StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)
            break

        pop_start_ls_res = get_pop_results(pop_start_ls, all_results_dict)
        pop_end_ls = copy.deepcopy(tuning_options.strategy_options["candidates"])
        pop_end_ls_res = get_pop_results(pop_end_ls, all_results_dict)
        afi_ls = calculate_afi(pop_start_ls_res, pop_end_ls_res, lsd, all_results_dict)

        # Adaptive Local Search Depth (ALSD)
        if afi_gs is not None and afi_ls is not None:
            if afi_ls > afi_gs:
                lsd += alsd
            elif afi_ls < afi_gs:
                lsd -= alsd
            # Less than 5 lsd doesn't make sense
            if lsd < 5:
                lsd = 5
            print(f"DEBUG: Adaptive Local Search Depth (ALSD) lsd = {lsd}", file=sys.stderr)

    ray.kill(cache_manager)
    for actor in actors:
        ray.kill(actor)

    return all_results

def calculate_afi(pop_before_rs, pop_after_rs, feval, results):
    # Average Fitness Increment (AFI)
    assert(feval >= 0)
    delta_fitness = fitness_increment(pop_before_rs, pop_after_rs)
    afi = delta_fitness / feval if feval > 0 else 0
    print(f"DEBUG:calculate_afi afi: {afi}", file=sys.stderr)
    return afi

def fitness_increment(pop_before, pop_after):
    if len(pop_before) != len(pop_after):
        raise ValueError("populations must have the same size.")
    
    sum_before = sum(t for t in pop_before if isinstance(t, float))
    sum_after = sum(t for t in pop_after if isinstance(t, float))
    difference_sum = sum_before - sum_after
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

def distribute_feval(feval, max_feval, maxiter, lsd, popsize, afi_gs, afi_ls):
    remaining_feval = max_feval - feval
    if remaining_feval < (lsd + maxiter) * popsize:
        # Calculate how many full batches of popsize can still be processed
        proportion = remaining_feval // popsize

        if afi_gs is None or afi_ls is None:
            maxiter = int(proportion * 0.5)
            lsd = int(proportion * 0.5)
        else:
            if afi_gs > afi_ls:
                # More evaluations to maxiter
                maxiter = int(proportion * 0.6)
                lsd = int(proportion * 0.4)
            else:
                # More evaluations to lsd
                maxiter = int(proportion * 0.4)
                lsd = int(proportion * 0.6)

        # If maxiter ends up being 1, assign all remaining feval to lsd
        if maxiter == 1:
            lsd = proportion  # Give all available batches to lsd
            maxiter = 0

        # Ensure at least one of maxiter or lsd is non-zero if there are still fevals to be used
        if maxiter == 0 and lsd == 0 and remaining_feval > 0:
            lsd = 1  # Allocate at least one batch to lsd to ensure progress

    return maxiter, lsd
 