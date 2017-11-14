""" The default runner for sequentially tuning the parameter space """
from __future__ import print_function

from collections import OrderedDict
import logging

from kernel_tuner.util import get_config_string
from kernel_tuner.core import DeviceInterface


class SequentialRunner(object):
    """ SequentialRunner is used for tuning with a single process/thread """

    def __init__(self, kernel_options, device_options, iterations):
        """ Instantiate the SequentialRunner

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
        self.dev = DeviceInterface(kernel_options.kernel_string, iterations=iterations, **device_options)

        #move data to the GPU
        self.gpu_args = self.dev.ready_argument_list(kernel_options.arguments)


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
            execution times. And a dictionary that contains a information
            about the hardware/software environment on which the tuning took place.
        :rtype: list(dict()), dict()

        """
        logging.debug('sequential runner started for ' + kernel_options.kernel_name)

        results = []

        #iterate over parameter space
        for element in parameter_space:
            params = OrderedDict(zip(tuning_options.tune_params.keys(), element))

            time = self.dev.compile_and_benchmark(self.gpu_args, params, kernel_options, tuning_options)

            if time is None:
                logging.debug('received time is None, kernel configuration was skipped silently due to compile or runtime failure')
                continue

            #print and append to results
            params['time'] = time
            output_string = get_config_string(params)
            logging.debug(output_string)
            print(output_string)
            results.append(params)

        return results, self.dev.get_environment()

    def __del__(self):
        if hasattr(self, 'dev'):
            del self.dev
