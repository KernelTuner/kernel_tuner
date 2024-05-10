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
from kernel_tuner.runners.parallel import ParallelRunner

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
    
    ensemble = []
    if "ensemble" in tuning_options:
        ensemble = tuning_options.ensemble
    else:
        ensemble = ["greedy_ils", "greedy_ils"]
    ensemble_size = len(ensemble)
    if num_devices < ensemble_size:
        warnings.warn("Number of devices is less than the number of strategies in the ensemble. Some strategies will wait until devices are available.", UserWarning)
    num_actors = num_devices if ensemble_size > num_devices else ensemble_size

    ensemble = [strategy_map[strategy] for strategy in ensemble]
    parallel_runner = ParallelRunner(runner.kernel_source, runner.kernel_options, runner.device_options, 
                                    runner.iterations, runner.observers, num_gpus=num_actors, cache_manager=cache_manager,
                                    simulation_mode=simulation_mode, actors=actors)
    final_results = parallel_runner.run(tuning_options=tuning_options, ensemble=ensemble, searchspace=searchspace)
    
    return final_results
