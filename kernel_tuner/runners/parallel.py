""" The default runner for sequentially tuning the parameter space """
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import warnings

from dask.distributed import Client

from kernel_tuner.util import get_config_string, store_cache
from kernel_tuner.core import DeviceInterface

from kernel_tuner.runners.sequential import SequentialRunner

class ParallelRunner(object):
    """ ParallelRunner is used for tuning with a remote Dask worker """

    def __init__(self, kernel_source, kernel_options, device_options, tuning_options, iterations, scheduler):
        """ Instantiate the RemoteRunner

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

        #connect to scheduler
        self.client = Client(scheduler)
        self.workers = self.client

        #this is here to get the number of workers
        probe = self.client.run(os.getcwd)
        self.nworkers = len(probe)

        #this is here for compatibility with the sequential runner
        self.dev = 1

        #store settings
        self.quiet = device_options.quiet
        self.kernel_source = kernel_source
        self.kernel_options = kernel_options
        self.device_options = device_options
        self.tuning_options = tuning_options
        self.iterations = iterations

        #args to create SeqentialRunners on the workers
        args = [kernel_source, kernel_options, device_options, tuning_options, iterations]

        #the following throws a warning, but since this is just for setup we can ignore it
        self.workers = []
        for key in probe.keys():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "", UserWarning)
                futures = self.client.submit(SequentialRunner, *args, actor=True, workers=key)
            self.workers.append(futures.result())
        #print(self.workers)

    def run(self, parameter_space):
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
        logging.debug('parallel runner started for ' + self.kernel_options.kernel_name)

        #this is to ensure we can subscript parameter_space later on
        parameter_space = list(parameter_space)
        futures = []

        #simply split the parameter space across the workers
        for i, worker in enumerate(self.workers):
            futures.append(worker.run(parameter_space[i::self.nworkers]))

        #now wait for all the futures, not the optimal way to do this, but anyway
        env = []
        results = []
        for future in futures:
            res, env = future.result()
            results += res

        #print to results unless quiet
        if not self.quiet:
            print_keys = list(self.tuning_options.tune_params.keys())+["time"]
            for d in results:
                output_string = get_config_string(d, print_keys, "")
                logging.debug(output_string)
                print(output_string)

        return results, env


    def __del__(self):
        if hasattr(self, 'dev'):
            del self.dev




