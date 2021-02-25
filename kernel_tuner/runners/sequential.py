""" The default runner for sequentially tuning the parameter space """
from __future__ import print_function

from collections import OrderedDict
import logging

from kernel_tuner.util import get_config_string, store_cache, process_metrics, print_config_output
from kernel_tuner.core import DeviceInterface


class SequentialRunner(object):
    """ SequentialRunner is used for tuning with a single process/thread """

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers):
        """ Instantiate the SequentialRunner

        :param kernel_source: The kernel source
        :type kernel_source: kernel_tuner.core.KernelSource

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param device_options: A dictionary with all options for the device
            on which the kernel should be tuned.
        :type device_options: kernel_tuner.interface.Options

        :param iterations: The number of iterations used for benchmarking
            each kernel instance.
        :type iterations: int
        """

        #detect language and create high-level device interface
        self.dev = DeviceInterface(kernel_source, iterations=iterations, observers=observers, **device_options).__enter__()

        self.units = self.dev.units
        self.quiet = device_options.quiet
        self.kernel_source = kernel_source

        self.warmed_up = False

        self.simulation_mode = False

        #move data to the GPU
        self.gpu_args = self.dev.ready_argument_list(kernel_options.arguments)

    def __enter__(self):
        return self

    def run(self, parameter_space, kernel_options, tuning_options):
        """ Iterate through the entire parameter space using a single Python process

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param tuning_options: A dictionary with all options regarding the tuning
            process.
        :type tuning_options: kernel_tuner.iterface.Options

        :returns: A list of dictionaries for executed kernel configurations and their
            execution times. And a dictionary that contains information
            about the hardware/software environment on which the tuning took place.
        :rtype: list(dict()), dict()

        """
        logging.debug('sequential runner started for ' + kernel_options.kernel_name)

        results = []

        #iterate over parameter space
        for element in parameter_space:
            params = OrderedDict(zip(tuning_options.tune_params.keys(), element))

            #attempt to warmup the GPU by running the first config in the parameter space and ignoring the result
            if not self.warmed_up:
                self.dev.compile_and_benchmark(self.kernel_source, self.gpu_args, params, kernel_options, tuning_options)
                self.warmed_up = True

            #check if element is in the cache
            x_int = ",".join([str(i) for i in element])
            if tuning_options.cache:
                if x_int in tuning_options.cache:
                    results.append(tuning_options.cache[x_int])
                    continue

            result = self.dev.compile_and_benchmark(self.kernel_source, self.gpu_args, params, kernel_options, tuning_options)
            if result is None:
                logging.debug('received benchmark result is None, kernel configuration was skipped silently due to compile or runtime failure')
                params.update({ "time": 1e20 })
                store_cache(x_int, params, tuning_options)
                continue

            #print and append to results
            if isinstance(result, dict):
                time = result["time"]
            else:
                time = result

            params['time'] = time

            if isinstance(result, dict):
                params.update(result)

            if tuning_options.metrics:
                params = process_metrics(params, tuning_options.metrics)

            print_config_output(tuning_options.tune_params, params, self.quiet, tuning_options.metrics, self.units)

            store_cache(x_int, params, tuning_options)
            results.append(params)

        return results, self.dev.get_environment()

    def __exit__(self, *exc):
        if hasattr(self, 'dev'):
            self.dev.__exit__(*exc)
