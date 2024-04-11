import logging
from datetime import datetime, timezone
from time import perf_counter
import ray
import sys

from kernel_tuner.util import ErrorConfig, print_config_output, process_metrics, store_cache
from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.sequential import SequentialRunner

@ray.remote(num_gpus=1)
class ParallelRemoteActor():
    def __init__(self, 
                 quiet,
                 kernel_source,
                 kernel_options, 
                 device_options,
                 iterations,
                 observers,
                 gpu_id):
        
        self.gpu_id = gpu_id
        self.dev = DeviceInterface(kernel_source, iterations=iterations, observers=observers, **device_options)
        self.units = self.dev.units
        self.quiet = quiet
        self.kernel_source = kernel_source
        self.warmed_up = False
        self.simulation_mode = False
        self.start_time = perf_counter()
        self.last_strategy_start_time = self.start_time
        self.last_strategy_time = 0
        self.kernel_options = kernel_options
        self.device_options = device_options
        self.iterations = iterations
        self.observers = observers
        self.cache_manager = None
        self.runner = None

    def execute(self, element, tuning_options):
        return self.runner.run([element], tuning_options)[0]

    def set_cache_manager(self, cache_manager):
        self.cache_manager = cache_manager

    def init_runner(self):
        if self.cache_manager is None:
            raise ValueError("Cache manager is not set.")
        self.runner = SequentialRunner(self.kernel_source, self.kernel_options, self.device_options, 
                                       self.iterations, self.observers, cache_manager=self.cache_manager)