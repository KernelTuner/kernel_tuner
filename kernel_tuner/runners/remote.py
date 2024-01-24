import logging
import ray
from ray.util.actor_pool import ActorPool

from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.runner import Runner
from kernel_tuner.runners.remote_actor import RemoteActor

class RemoteRunner(Runner):

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers):
        #detect language and create high-level device interface
        self.dev = DeviceInterface(kernel_source, iterations=iterations, observers=observers, **device_options)
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
        # Create a pool of RemoteActor actors
        self.actor_pool = ActorPool(self.actors)

    def get_environment(self, tuning_options):
        return self.dev.get_environment()

    
    def run(self, parameter_space, tuning_options):
        results = list(self.actor_pool.map_unordered(lambda a, v: a.execute.remote(v, tuning_options), parameter_space))
        return results
    