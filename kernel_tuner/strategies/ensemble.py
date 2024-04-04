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
from kernel_tuner.runners.remote_actor import RemoteActor
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
    # Define cluster resources
    num_gpus = get_num_devices(runner.kernel_source.lang)
    print(f"Number of GPUs in use: {num_gpus}", file=sys. stderr)
    resources = {}
    for id in range(num_gpus):
        gpu_resource_name = f"gpu_{id}"
        resources[gpu_resource_name] = 1
    # Initialize Ray
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(resources=resources, include_dashboard=True)
    # Create RemoteActor instances
    actors = [create_actor_on_gpu(id, runner) for id in range(num_gpus)]
    # Create a pool of RemoteActor actors
    #actor_pool = ActorPool(actors)
    
    if "ensemble" in tuning_options:
        ensemble = tuning_options["ensemble"]
    else:
        ensemble = ["random_sample", "random_sample", "random_sample"] # For now its just a random ensemble not based on any logic
    
    ensemble = [strategy_map[strategy] for strategy in ensemble]
    tasks = []
    simulation_mode = True if isinstance(runner, SimulationRunner) else False
    for i in range(len(ensemble)):
        strategy = ensemble[i]
        actor = actors[i]
        task = actor.execute.remote(strategy, searchspace, tuning_options, simulation_mode)
        tasks.append(task)
    all_results = ray.get(tasks)

    unique_configs = set()
    final_results = []

    for strategy_results in all_results:
        for new_result in strategy_results:
            config_signature = tuple(new_result[param] for param in searchspace.tune_params)

            if config_signature not in unique_configs:
                final_results.append(new_result)
                unique_configs.add(config_signature)

    return final_results

# ITS REPEATING CODE, SAME IN parallel.py
def create_actor_on_gpu(gpu_id, runner):
    gpu_resource_name = f"gpu_{gpu_id}"
    return RemoteActor.options(resources={gpu_resource_name: 1}).remote(runner.quiet,
                                                                        runner.kernel_source, 
                                                                        runner.kernel_options, 
                                                                        runner.device_options, 
                                                                        runner.iterations, 
                                                                        runner.observers,
                                                                        gpu_id)