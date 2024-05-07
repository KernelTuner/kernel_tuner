import logging
import ray
import sys
import os
from ray.util.actor_pool import ActorPool
from time import perf_counter

from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.runner import Runner
from kernel_tuner.runners.ray.remote_actor import RemoteActor
from kernel_tuner.util import get_num_devices
from kernel_tuner.runners.ray.cache_manager import CacheManager
from kernel_tuner.strategies.common import create_actor_on_device, initialize_ray

class ParallelRunner(Runner):

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers, 
                 num_gpus=None, cache_manager=None, actors=None, simulation_mode=False):
        self.dev = DeviceInterface(kernel_source, iterations=iterations, observers=observers, **device_options) if not simulation_mode else None
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
        self.actors = actors

        if num_gpus is None:
            self.num_gpus = get_num_devices(kernel_source.lang, simulation_mode=self.simulation_mode)

        initialize_ray(num_gpus)

        # Create RemoteActor instances
        if actors is None:
            runner_attributes = [self.kernel_source, self.kernel_options, self.device_options, self.iterations, self.observers]
            self.actors = [create_actor_on_device(*runner_attributes, self.cache_manager, simulation_mode, id) for id in range(self.num_gpus)]

    def get_environment(self, tuning_options):
        return self.dev.get_environment()

    
    def run(self, parameter_space, tuning_options, cache_manager=None):
        if self.cache_manager is None:
            if cache_manager is None:
                cache_manager = CacheManager.remote(tuning_options)
            self.cache_manager = cache_manager
        # set the cache manager for each actor. Can't be done in constructor because we do not have yet the tuning_options
        for actor in self.actors:
            ray.get(actor.set_cache_manager.remote(self.cache_manager))
        # Create a pool of RemoteActor actors
        self.actor_pool = ActorPool(self.actors)
        # Distribute execution of the `execute` method across the actor pool with varying parameters and tuning options, collecting the results asynchronously.
        results = list(self.actor_pool.map_unordered(lambda a, v: a.execute.remote(element=v, tuning_options=tuning_options), parameter_space))
        new_tuning_options = ray.get(self.cache_manager.get_tuning_options.remote())
        tuning_options.update(new_tuning_options)
        
        return results
