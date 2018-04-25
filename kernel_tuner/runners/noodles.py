""" The Noodles runner allows tuning in parallel using multiple processes/threads """
import subprocess
import random
from collections import OrderedDict

import numpy

from noodles import schedule_hint, gather, lift
from noodles.run.runners import run_parallel_with_display, run_parallel
from noodles.display import NCDisplay

from kernel_tuner.core import DeviceInterface

def _error_filter(errortype, value=None, tb=None):
    if errortype is subprocess.CalledProcessError:
        return value.stderr
    elif "cuCtxSynchronize" in str(value):
        return value
    return None

def _chunk_list(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class NoodlesRunner:

    def __init__(self, device_options, max_threads=1):
        self.device_options = device_options
        self.device_options["quiet"] = True
        self.max_threads = max_threads
        self.dev = None

    def run(self, parameter_space, kernel_options, tuning_options):
        """ Tune all instances in parameter_space using a multiple threads

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param tuning_options: A dictionary with all options regarding the tuning
            process.
        :type tuning_options: kernel_tuner.interface.Options

        :returns: A list of dictionaries for executed kernel configurations and their
            execution times. And a dictionary that contains a information
            about the hardware/software environment on which the tuning took place.
        :rtype: list(dict()), dict()

        """
        workflow = self._parameter_sweep(parameter_space, kernel_options, self.device_options,
                                         tuning_options)
        if tuning_options.verbose:
            with NCDisplay(_error_filter) as display:
                answer = run_parallel_with_display(workflow, self.max_threads, display)
        else:
            answer = run_parallel(workflow, self.max_threads)

        if answer is None:
            print("Tuning did not return any results, did an error occur?")
            return None

        # Filter out None times
        result = []
        for chunk in answer:
            result += [d for d in chunk if d['time']]

        return result, {}


    @schedule_hint(display="Batching ... ",
                   ignore_error=True,
                   confirm=True)
    def _parameter_sweep(self, parameter_space, kernel_options, device_options, tuning_options):
        """Build a Noodles workflow by sweeping the parameter space"""
        results = []

        #randomize parameter space to do pseudo load balancing
        parameter_space = list(parameter_space)
        random.shuffle(parameter_space)

        #split parameter space into chunks
        work_per_thread = int(numpy.ceil(len(parameter_space) / float(self.max_threads)))
        chunks = _chunk_list(parameter_space, work_per_thread)

        for chunk in chunks:

            chunked_result = self._run_chunk(chunk, kernel_options, device_options, tuning_options)

            results.append(lift(chunked_result))

        return gather(*results)


    @schedule_hint(ignore_error=True,
                   confirm=True)
    def _run_chunk(self, chunk, kernel_options, device_options, tuning_options):
        """Benchmark a single kernel instance in the parameter space"""

        #detect language and create high-level device interface
        self.dev = DeviceInterface(kernel_options.kernel_string, iterations=tuning_options.iterations, **device_options)

        #move data to the GPU
        gpu_args = self.dev.ready_argument_list(kernel_options.arguments)

        results = []

        for element in chunk:
            params = dict(OrderedDict(zip(tuning_options.tune_params.keys(), element)))

            try:
                time = self.dev.compile_and_benchmark(gpu_args, params, kernel_options, tuning_options)

                params['time'] = time
                results.append(params)
            except Exception:
                params['time'] = None
                results.append(params)

        return results
