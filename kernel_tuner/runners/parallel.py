import logging
import ray
import sys
import os
from ray.util.actor_pool import ActorPool
from time import perf_counter
from collections import deque
import copy

from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.runner import Runner
from kernel_tuner.runners.ray.remote_actor import RemoteActor
from kernel_tuner.util import get_num_devices, get_nested_types
from kernel_tuner.runners.ray.cache_manager import CacheManager
from kernel_tuner.strategies.common import create_actor_on_device, initialize_ray

class ParallelRunner(Runner):

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers, 
                 num_gpus=None, cache_manager=None, actors=None, simulation_mode=False):
        self.dev = DeviceInterface(kernel_source, iterations=iterations, observers=observers, **device_options) if not simulation_mode else None
        self.quiet = device_options.quiet
        self.kernel_source = kernel_source
        self.warmed_up = False
        self.simulation_mode = simulation_mode
        self.start_time = perf_counter()
        self.last_strategy_start_time = self.start_time
        self.last_strategy_time = 0
        self.kernel_options = kernel_options
        self.observers = observers
        self.iterations = iterations
        self.device_options = device_options
        self.cache_manager = cache_manager
        self.num_gpus = num_gpus
        self.actors = actors

        if num_gpus is None:
            self.num_gpus = get_num_devices(kernel_source.lang, simulation_mode=self.simulation_mode)

        initialize_ray()

        # Create RemoteActor instances
        if actors is None:
            runner_attributes = [self.kernel_source, self.kernel_options, self.device_options, self.iterations, self.observers]
            self.actors = [create_actor_on_device(*runner_attributes, self.cache_manager, simulation_mode, id) for id in range(self.num_gpus)]

    def get_environment(self, tuning_options):
        return self.dev.get_environment()

    
    def run(self, parameter_space=None, tuning_options=None, ensemble=None, searchspace=None, cache_manager=None):
        if tuning_options is None: #HACK as tuning_options can't be the first argument and parameter_space needs to be a default argument
            raise ValueError("tuning_options cannot be None")
        
        if self.cache_manager is None:
            if cache_manager is None:
                cache_manager = CacheManager.remote(tuning_options)
            self.cache_manager = cache_manager
        
        # set the cache manager for each actor. Can't be done in constructor because we do not always yet have the tuning_options
        for actor in self.actors:
            ray.get(actor.set_cache_manager.remote(self.cache_manager))
    
        # Determine what type of parallelism and run appropriately
        if parameter_space and not ensemble and not searchspace:
            results, tuning_options_list = self.run_parallel_tuning(tuning_options, parameter_space)
        elif ensemble and searchspace and not parameter_space:
            results, tuning_options_list = self.run_parallel_ensemble(ensemble, tuning_options, searchspace)
        else:
            raise ValueError("Invalid arguments to parallel runner run method")
        
        # Update tuning options
        new_tuning_options = ray.get(self.cache_manager.get_tuning_options.remote())
        tuning_options.update(new_tuning_options)
        if self.simulation_mode:
            tuning_options.simulated_time += self._calculate_simulated_time(tuning_options_list)
            print(f"DEBUG: simulated_time = {tuning_options.simulated_time}", file=sys.stderr)
        
        return results

    def run_parallel_ensemble(self, ensemble, tuning_options, searchspace):
        """
        Runs strategies from the ensemble in parallel using distributed actors, 
        manages dynamic task allocation, and collects results.
        """
        ensemble_queue = deque(ensemble)
        pending_tasks = {}
        all_results = []

        # Start initial tasks for each actor
        for actor in self.actors:
            strategy = ensemble_queue.popleft()
            remote_tuning_options = self._setup_tuning_options(tuning_options)
            task = actor.execute.remote(strategy=strategy, searchspace=searchspace, tuning_options=remote_tuning_options)
            pending_tasks[task] = actor
        
        # Manage task completion and redistribution
        while pending_tasks:
            done_ids, _ = ray.wait(list(pending_tasks.keys()), num_returns=1)
            for done_id in done_ids:
                result = ray.get(done_id)
                all_results.append(result)
                actor = pending_tasks.pop(done_id)

                # Reassign actors if strategies remain
                if ensemble_queue:
                    strategy = ensemble_queue.popleft()
                    remote_tuning_options = self._setup_tuning_options(tuning_options)
                    task = actor.execute.remote(strategy=strategy, searchspace=searchspace, tuning_options=remote_tuning_options)
                    pending_tasks[task] = actor
        
        # Process results to extract population and candidates for further use
        results, tuning_options_list, population, candidates = self._process_results_ensemble(all_results)

        # Update tuning options for memetic strategies
        if population:
            tuning_options.strategy_options["population"] = population
        if candidates:
            tuning_options.strategy_options["candidates"] = candidates
        return results, tuning_options_list
    
    def _setup_tuning_options(self, tuning_options):
        new_tuning_options = copy.deepcopy(tuning_options)
        if "candidates" in tuning_options.strategy_options:
            if len(tuning_options.strategy_options["candidates"]) > 0:
                new_tuning_options.strategy_options["candidate"] = tuning_options.strategy_options["candidates"].pop(0)
        return new_tuning_options
    
    def _process_results_ensemble(self, all_results):
        population = [] # for memetic strategy
        candidates = [] # for memetic strategy
        results = []
        tuning_options_list = []

        for (strategy_results, tuning_options) in all_results:
            if "old_candidate" in tuning_options.strategy_options:
                candidates.append(tuning_options.strategy_options["old_candidate"])
            if "candidate" in tuning_options.strategy_options:
                population.append(tuning_options.strategy_options["candidate"])
            results.extend(strategy_results)
            tuning_options_list.append(tuning_options)

        return results, tuning_options_list, population, candidates


    def run_parallel_tuning(self, tuning_options, parameter_space):
        # Create a pool of RemoteActor actors
        self.actor_pool = ActorPool(self.actors)
        # Distribute execution of the `execute` method across the actor pool with varying parameters and tuning options, collecting the results asynchronously.
        all_results = list(self.actor_pool.map_unordered(lambda a, v: a.execute.remote(element=v, tuning_options=tuning_options), parameter_space))
        results = [x[0] for x in all_results]
        tuning_options_list = [x[1] for x in all_results]
        return results, tuning_options_list
    
    def _process_results(self, all_results, searchspace):
        unique_configs = set()
        final_results = []

        for (strategy_results, tuning_options) in all_results:
            for new_result in strategy_results:
                config_signature = tuple(new_result[key] for key in searchspace.tune_params)
                if config_signature not in unique_configs:
                    final_results.append(new_result)
                    unique_configs.add(config_signature)
        return final_results
    
    def _calculate_simulated_time(self, tuning_options_list):
        simulated_times = []
        for tuning_options in tuning_options_list:
            print(f"DEBUG:_calculate_simulated_time tuning_options.simulated_time = {tuning_options.simulated_time}", file=sys.stderr)
            simulated_times.append(tuning_options.simulated_time)
        #simulated_times = [tuning_options.simulated_time for tuning_options in tuning_options_list]
        return max(simulated_times)