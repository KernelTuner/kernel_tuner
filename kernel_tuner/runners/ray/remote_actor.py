import logging
from datetime import datetime, timezone
from time import perf_counter
import ray
import sys

from kernel_tuner.util import ErrorConfig, print_config_output, process_metrics, store_cache
from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner.runners.simulation import SimulationRunner

@ray.remote
class RemoteActor():
    def __init__(self, 
                 kernel_source,
                 kernel_options, 
                 device_options,
                 iterations,
                 observers,
                 cache_manager):
        
        self.kernel_source = kernel_source
        self.simulation_mode = False
        self.kernel_options = kernel_options
        self.device_options = device_options
        self.iterations = iterations
        self.observers = observers
        self.cache_manager = cache_manager
        self.runner = None
        
    def execute(self, strategy, searchspace, tuning_options, simulation_mode=False):
        if simulation_mode:
            self.runner = SimulationRunner(self.kernel_source, self.kernel_options, self.device_options, 
                                 self.iterations, self.observers)
        else:
            self.runner = SequentialRunner(self.kernel_source, self.kernel_options, self.device_options, 
                                 self.iterations, self.observers, cache_manager=self.cache_manager)
        results = strategy.tune(searchspace, self.runner, tuning_options)
        return results, tuning_options
    