import random
import sys
import os
import ray
from ray.util.actor_pool import ActorPool

import numpy as np

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc, scale_from_params
from kernel_tuner.runners.simulation import SimulationRunner
from kernel_tuner.runners.ray.remote_actor import RemoteActor
from kernel_tuner.util import get_num_devices
from kernel_tuner.runners.ray.cache_manager import CacheManager

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

def tune(searchspace: Searchspace, runner, tuning_options):
    simulation_mode = True if isinstance(runner, SimulationRunner) else False
    if "ensemble" in tuning_options:
        ensemble = tuning_options["ensemble"]
    else:
        ensemble = ["random_sample", "random_sample"]

    # Define cluster resources
    num_devices = get_num_devices(runner.kernel_source.lang, simulation_mode=simulation_mode)
    print(f"Number of devices available: {num_devices}", file=sys. stderr)
    if num_devices < len(ensemble):
        raise ValueError(f"Number of devices ({num_devices}) is less than the number of strategies in the ensemble ({len(ensemble)})")
    
    resources = {}
    for id in range(len(ensemble)):
        device_resource_name = f"gpu_{id}"
        resources[device_resource_name] = 1
    resources["cache_manager_cpu"] = 1
    # Initialize Ray
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(resources=resources, include_dashboard=True, ignore_reinit_error=True)
    # Create cache manager and actors
    cache_manager = CacheManager.options(resources={"cache_manager_cpu": 1}).remote(tuning_options)
    actors = [create_actor_on_device(id, runner, cache_manager, simulation_mode) for id in range(len(ensemble))]
    
    ensemble = [strategy_map[strategy] for strategy in ensemble]
    tasks = []
    for i in range(len(ensemble)):
        strategy = ensemble[i]
        actor = actors[i]
        task = actor.execute.remote(strategy, searchspace, tuning_options, simulation_mode)
        tasks.append(task)
    all_results = ray.get(tasks)
    tuning_options = ray.get(cache_manager.get_tuning_options.remote())

    unique_configs = set()
    final_results = []

    for strategy_results in all_results:
        for new_result in strategy_results:
            config_signature = tuple(new_result[param] for param in searchspace.tune_params)

            if config_signature not in unique_configs:
                final_results.append(new_result)
                unique_configs.add(config_signature)

    #kill all actors and chache manager
    for actor in actors:
        ray.kill(actor)
    ray.kill(cache_manager)

    return final_results

def create_actor_on_device(gpu_id, runner, cache_manager, simulation_mode):
    gpu_resource_name = f"gpu_{gpu_id}"
    if simulation_mode:
        resource_options= {"num_cpus": 1}
    else:
        resource_options= {"num_gpus": 1}
    return RemoteActor.options(**resource_options, resources={gpu_resource_name: 1}).remote(runner.kernel_source, 
                                                                                            runner.kernel_options, 
                                                                                            runner.device_options, 
                                                                                            runner.iterations, 
                                                                                            runner.observers,
                                                                                            cache_manager)