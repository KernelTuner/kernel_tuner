import random
import sys
import os
import ray
import copy
import logging
import warnings
from collections import deque

import numpy as np

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc, scale_from_params, check_num_devices, create_actor_on_device, initialize_ray
from kernel_tuner.runners.simulation import SimulationRunner
from kernel_tuner.runners.ray.remote_actor import RemoteActor
from kernel_tuner.util import get_num_devices
from kernel_tuner.runners.ray.cache_manager import CacheManager

from kernel_tuner.strategies import (
    basinhopping,
    bayes_opt,
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
)

strategy_map = {
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

def tune(searchspace: Searchspace, runner, tuning_options, cache_manager=None, actors=None):
    simulation_mode = True if isinstance(runner, SimulationRunner) else False
    num_devices = get_num_devices(runner.kernel_source.lang, simulation_mode=simulation_mode)
    print(f"DEBUG: num_devices={num_devices}", file=sys.stderr)
    
    ensemble = []
    if "ensemble" in tuning_options:
        ensemble = tuning_options["ensemble"]
    else:
        ensemble = ["greedy_ils", "greedy_ils"]
    ensemble_size = len(ensemble)
    if num_devices < ensemble_size:
        warnings.warn("Number of devices is less than the number of strategies in the ensemble. Some strategies will wait until devices are available.", UserWarning)
    num_actors = num_devices if ensemble_size > num_devices else ensemble_size
    
    initialize_ray(num_devices)
    
    # Create cache manager and actors
    if cache_manager is None:
        cache_manager = CacheManager.remote(tuning_options)
    if actors is None:
        runner_attributes = [runner.kernel_source, runner.kernel_options, runner.device_options, runner.iterations, runner.observers]
        actors = [create_actor_on_device(*runner_attributes, cache_manager, simulation_mode, id) for id in range(num_actors)]
    
    # Execute all actor with one strategy each
    ensemble = [strategy_map[strategy] for strategy in ensemble]
    ensemble_queue = deque(ensemble)
    pending_tasks = {}
    for actor in actors:
        strategy = ensemble_queue.popleft()
        remote_tuning_options = setup_tuning_options(tuning_options)
        task = actor.execute.remote(strategy=strategy, searchspace=searchspace, tuning_options=remote_tuning_options)
        pending_tasks[task] = actor
    

    all_results = []
    while pending_tasks:
        done_ids, _ = ray.wait(list(pending_tasks.keys()), num_returns=1)
        for done_id in done_ids:
            result = ray.get(done_id)
            all_results.append(result)
            actor = pending_tasks.pop(done_id)

            if ensemble_queue:
                strategy = ensemble_queue.popleft()
                remote_tuning_options = setup_tuning_options(tuning_options)
                task = actor.execute.remote(strategy=strategy, searchspace=searchspace, tuning_options=remote_tuning_options)
                pending_tasks[task] = actor

    new_tuning_options = ray.get(cache_manager.get_tuning_options.remote())
    tuning_options.update(new_tuning_options)
    final_results, population, candidates = process_results(all_results, searchspace)

    if population: # for memetic strategy
        tuning_options.strategy_options["population"] = population
    if candidates: # for memetic strategy
        tuning_options.strategy_options["candidates"] = candidates

    return final_results

def setup_tuning_options(tuning_options):
    new_tuning_options = copy.deepcopy(tuning_options)
    if "candidates" in tuning_options.strategy_options:
        if len(tuning_options.strategy_options["candidates"]) > 0:
            new_tuning_options.strategy_options["candidate"] = tuning_options.strategy_options["candidates"].pop(0)
    return new_tuning_options

def process_results(all_results, searchspace):
    unique_configs = set()
    final_results = []
    population = [] # for memetic strategy
    candidates = [] # for memetic strategy

    for (strategy_results, tuning_options) in all_results:
        if "old_candidate" in tuning_options.strategy_options:
            candidates.append(tuning_options.strategy_options["old_candidate"])
        if "candidate" in tuning_options.strategy_options:
            population.append(tuning_options.strategy_options["candidate"])
        for new_result in strategy_results:
            config_signature = tuple(new_result[key] for key in searchspace.tune_params)
            if config_signature not in unique_configs:
                final_results.append(new_result)
                unique_configs.add(config_signature)
    return final_results, population, candidates
