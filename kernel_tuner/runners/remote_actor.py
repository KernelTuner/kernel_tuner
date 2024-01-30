import logging
from datetime import datetime, timezone
from time import perf_counter
import ray
import sys

from kernel_tuner.util import ErrorConfig, print_config_output, process_metrics, store_cache
from kernel_tuner.core import DeviceInterface

@ray.remote(num_gpus=1)
class RemoteActor:
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
        #move data to the GPU
        self.gpu_args = self.dev.ready_argument_list(self.kernel_options.arguments)

    def execute(self, element, tuning_options):
        #print(f"GPU {self.gpu_id} started execution", file=sys. stderr)
        params = dict(zip(tuning_options.tune_params.keys(), element))

        result = None
        warmup_time = 0

        # check if configuration is in the cache
        x_int = ",".join([str(i) for i in element])
        if tuning_options.cache and x_int in tuning_options.cache:
            params.update(tuning_options.cache[x_int])
            params['compile_time'] = 0
            params['verification_time'] = 0
            params['benchmark_time'] = 0
        else:
            # attempt to warmup the GPU by running the first config in the parameter space and ignoring the result
            if not self.warmed_up:
                warmup_time = perf_counter()
                self.dev.compile_and_benchmark(self.kernel_source, self.gpu_args, params, self.kernel_options, tuning_options)
                self.warmed_up = True
                warmup_time = 1e3 * (perf_counter() - warmup_time)

            result = self.dev.compile_and_benchmark(self.kernel_source, self.gpu_args, params, self.kernel_options, tuning_options)

            params.update(result)

            if tuning_options.objective in result and isinstance(result[tuning_options.objective], ErrorConfig):
                logging.debug('kernel configuration was skipped silently due to compile or runtime failure')

        # only compute metrics on configs that have not errored
        if tuning_options.metrics and not isinstance(params.get(tuning_options.objective), ErrorConfig):
            params = process_metrics(params, tuning_options.metrics)

        # get the framework time by estimating based on other times
        total_time = 1000 * (perf_counter() - self.start_time) - warmup_time
        params['strategy_time'] = self.last_strategy_time
        params['framework_time'] = max(total_time - (params['compile_time'] + params['verification_time'] + params['benchmark_time'] + params['strategy_time']), 0)
        params['timestamp'] = str(datetime.now(timezone.utc))
        self.start_time = perf_counter()

        if result:
            # print configuration to the console
            print_config_output(tuning_options.tune_params, params, self.quiet, tuning_options.metrics, self.units)

            # add configuration to cache
            store_cache(x_int, params, tuning_options)

        return params