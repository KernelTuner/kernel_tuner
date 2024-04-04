import logging
from datetime import datetime, timezone
from time import perf_counter
import ray
import sys

from kernel_tuner.util import ErrorConfig, print_config_output, process_metrics, store_cache
from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner.runners.simulation import SimulationRunner

@ray.remote(num_gpus=1)
class RemoteActor():
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
        #move data to the GPU
        self.gpu_args = self.dev.ready_argument_list(self.kernel_options.arguments)

    def execute(self, strategy, searchspace, tuning_options, simulation_mode=False):
        selected_runner = SimulationRunner if simulation_mode else SequentialRunner
        runner = selected_runner(self.kernel_source, self.kernel_options, self.device_options, 
                                 self.iterations, self.observers)
        results = strategy.tune(searchspace, runner, tuning_options)
        return results