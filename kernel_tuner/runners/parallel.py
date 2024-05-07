import logging
import ray
import sys
import os
from ray.util.actor_pool import ActorPool
from time import perf_counter

from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.runner import Runner
from kernel_tuner.runners.ray.parallel_remote_actor import ParallelRemoteActor
from kernel_tuner.util import get_num_devices

class ParallelRunner(Runner):

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers, num_gpus, cache_manager=None):
        self.dev = DeviceInterface(kernel_source, iterations=iterations, observers=observers, **device_options)
        self.units = self.dev.units
        self.quiet = device_options.quiet
        self.kernel_source = kernel_source
        self.warmed_up = False
        self.simulation_mode = False
        self.start_time = perf_counter()
        self.last_strategy_start_time = self.start_time
        self.last_strategy_time = 0
        self.kernel_options = kernel_options
        self.observers = observers
        self.iterations = iterations
        self.device_options = device_options
        self.cache_manager = cache_manager
        self.num_gpus = num_gpus

        # Initialize Ray
        if not ray.is_initialized():
            os.environ["RAY_DEDUP_LOGS"] = "0"
            ray.init(include_dashboard=True, ignore_reinit_error=True)

    def get_environment(self, tuning_options):
        return self.dev.get_environment()

    
    def run(self, parameter_space, tuning_options, cache_manager=None):
        if self.cache_manager is None:
            if cache_manager is None:
                raise ValueError("A cache manager is required for parallel execution")
            self.cache_manager = cache_manager
        # Create RemoteActor instances
        self.actors = [self.create_actor_on_gpu(self.cache_manager) for _ in range(self.num_gpus)]
        # Create a pool of RemoteActor actors
        self.actor_pool = ActorPool(self.actors)
        # Distribute execution of the `execute` method across the actor pool with varying parameters and tuning options, collecting the results asynchronously.
        results = list(self.actor_pool.map_unordered(lambda a, v: a.execute.remote(v, tuning_options), parameter_space))
        new_tuning_options = ray.get(self.cache_manager.get_tuning_options.remote())
        tuning_options.update(new_tuning_options)

        for actor in self.actors:
            ray.kill(actor)
        
        return results
    
    def create_actor_on_gpu(self, cache_manager):
        return ParallelRemoteActor.remote(self.quiet,
                                            self.kernel_source, 
                                            self.kernel_options, 
                                            self.device_options, 
                                            self.iterations, 
                                            self.observers,
                                            cache_manager)