import logging
import ray
import sys
import os
from ray.util.actor_pool import ActorPool
from time import perf_counter

from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.runner import Runner
from kernel_tuner.runners.remote_actor import RemoteActor

class RemoteRunner(Runner):

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers):
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

        # Define cluster resources
        self.num_gpus = get_num_devices(kernel_source.lang)
        print(f"Number of GPUs in use: {self.num_gpus}", file=sys. stderr)
        resources = {}
        for id in range(self.num_gpus):
            gpu_resource_name = f"gpu_{id}"
            resources[gpu_resource_name] = 1
        # Initialize Ray
        os.environ["RAY_DEDUP_LOGS"] = "0"
        ray.init(resources=resources, include_dashboard=True)
        # Create RemoteActor instances
        self.actors = [self.create_actor_on_gpu(id) for id in range(self.num_gpus)]
        # Create a pool of RemoteActor actors
        self.actor_pool = ActorPool(self.actors)

    def get_environment(self, tuning_options):
        return self.dev.get_environment()

    
    def run(self, parameter_space, tuning_options):
        print(f"Size parameter_space: {len(parameter_space)}", file=sys. stderr)
        results = list(self.actor_pool.map_unordered(lambda a, v: a.execute.remote(v, tuning_options), parameter_space))
        return results
    
    def create_actor_on_gpu(self, gpu_id):
        gpu_resource_name = f"gpu_{gpu_id}"
        return RemoteActor.options(resources={gpu_resource_name: 1}).remote(self.quiet,
                                                                            self.kernel_source, 
                                                                            self.kernel_options, 
                                                                            self.device_options, 
                                                                            self.iterations, 
                                                                            self.observers,
                                                                            gpu_id)

# DONT KNOW WHERE TO PUT IT YET
def get_num_devices(lang):
    num_devices = 0
    if lang.upper() == "CUDA":
        import pycuda.driver as cuda
        cuda.init()
        num_devices = cuda.Device.count()
    elif lang.upper() == "CUPY":
        import cupy
        num_devices = cupy.cuda.runtime.getDeviceCount()
    elif lang.upper() == "NVCUDA":
        # NVCUDA usually refers to NVIDIA's CUDA, so you can use pycuda or a similar approach
        import pycuda.driver as cuda
        cuda.init()
        num_devices = cuda.Device.count()
    elif lang.upper() == "OPENCL":
        import pyopencl as cl
        num_devices = sum(len(platform.get_devices()) for platform in cl.get_platforms())
    elif lang.upper() == "HIP":
        from pyhip import hip
        num_devices = hip.hipGetDeviceCount()
    else:
        raise ValueError(f"Unsupported language: {lang}")

    return num_devices