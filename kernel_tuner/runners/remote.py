import logging
from datetime import datetime, timezone
from time import perf_counter
import ray

from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.runner import Runner
from kernel_tuner.runners.remote_actor import RemoteActor
from kernel_tuner.util import ErrorConfig, print_config_output, process_metrics, store_cache


class RemoteRunner(Runner):

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers):
        #detect language and create high-level device interface
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

        # Initialize Ray
        ray.init(ignore_reinit_error=True)

        # Get cluster resources
        cluster_resources = ray.cluster_resources()
        self.num_gpus = int(cluster_resources.get("GPU", 0))  # Default to 0 if no GPUs are found

        # Create RemoteActor instances
        self.actors = [RemoteActor.remote(self.dev.units, 
                                          device_options.quiet,
                                          kernel_source, 
                                          kernel_options, 
                                          device_options, 
                                          iterations, 
                                          observers) for _ in range(self.num_gpus)]

    def get_environment(self, tuning_options):
        return self.dev.get_environment()

    
    def run(self, parameter_space, tuning_options):
        future_results = []

        # Iterate over parameter space and distribute work to actors
        for element in parameter_space:
            future = [actor.execute.remote(element, tuning_options) for actor in self.actors]
            future_results.extend(future)

        return ray.get(future_results)
    