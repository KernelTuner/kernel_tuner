""" This module contains the functionality for running Python functions """

from collections import namedtuple
import platform
import logging
import warnings
import importlib.util
from math import ceil
from time import perf_counter
from typing import Tuple

# import cProfile

import progressbar
import numpy as np

# for parallel subprocess runs
from multiprocess import Manager, cpu_count, get_context    # using Pathos as Python's multiprocessing is unable to pickle
from itertools import repeat
import subprocess
import sys
from os import getpid

from kernel_tuner.util import get_temp_filename, delete_temp_file

# This represents an individual kernel argument.
# It contains a numpy object (ndarray or number) and a ctypes object with a copy
# of the argument data. For an ndarray, the ctypes object is a wrapper for the ndarray's data.
Argument = namedtuple("Argument", ["numpy", "ctypes"])
invalid_value = 1e20


class PythonFunctions(object):
    """Class that groups the code for running and compiling C functions"""

    def __init__(self, iterations=7, observers=None, parallel_mode=False, show_progressbar=False):
        """instantiate PythonFunctions object used for interacting with Python code

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int
        """
        self.iterations = iterations
        self.max_threads = 1024
        self.show_progressbar = show_progressbar

        #environment info
        env = dict()
        env["iterations"] = self.iterations
        self.env = env
        self.name = platform.processor()
        self.observers = observers or []
        self.parallel_mode = parallel_mode

        self.benchmark_times = []

        if self.parallel_mode:
            warnings.warn(
                "Be sure to check that simulation mode is true for the kernel, because parallel mode requires a completed cache file to avoid race conditions.")

        if len(self.observers) > 0 and self.parallel_mode:
            raise NotImplementedError("Observers are currently not implemented for parallel execution.")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the Python function
        """
        return arguments

    def compile(self, kernel_instance):
        """ return the function from the kernel instance """

        suffix = kernel_instance.kernel_source.get_user_suffix()
        source_file = get_temp_filename(suffix=suffix)

        spec = importlib.util.find_spec(kernel_instance.name)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        func = getattr(foo, kernel_instance.name)

        self.params = kernel_instance.params

        delete_temp_file(source_file)
        return func

    def benchmark(self, func, args, threads, grid):
        """runs the kernel repeatedly, returns averaged returned value

        The C function tuning is a little bit more flexible than direct CUDA
        or OpenCL kernel tuning. The C function needs to measure time, or some
        other quality metric you wish to tune on, on its own and should
        therefore return a single floating-point value.

        Benchmark runs the C function repeatedly and returns the average of the
        values returned by the C function. The number of iterations is set
        during the creation of the CFunctions object. For all measurements the
        lowest and highest values are discarded and the rest is included in the
        average. The reason for this is to be robust against initialization
        artifacts and other exceptional cases.

        :param func: A C function compiled for this specific configuration
        :type func: ctypes._FuncPtr

        :param args: A list of arguments to the function, order should match the
            order in the code. The list should be prepared using
            ready_argument_list().
        :type args: list(Argument)

        :param threads: Ignored, but left as argument for now to have the same
            interface as CudaFunctions and OpenCLFunctions.
        :type threads: any

        :param grid: Ignored, but left as argument for now to have the same
            interface as CudaFunctions and OpenCLFunctions.
        :type grid: any

        :returns: All execution times.
        :rtype: dict()
        """

        # For reference: the following times were obtained with 35 repeats on random_sample strategy.
        # As seen, there is a lot of overhead with subproceses; directly executing the function scales much better.
        # time taken by sequential: 20.7 sec
        # time taken by parallel in sequential form (subprocess overhead): 46.3 sec
        # time taken by parallel subprocesses: 7.5 sec on 9, 9.9 sec on 8, 13.6 sec on 4, 27.8 sec on 2, 45.9 sec on 1
        # time taken by parallel directly: 2.99 sec on 9, 4.0 sec on 8, 5.23 sec on 4, 11.3 sec on 2, 19.3 sec on 1

        result = dict()
        result["times"] = []
        min_valid_iterations = ceil(self.iterations * 0.8)
        iterator = range(self.iterations) if not self.show_progressbar or self.parallel_mode else progressbar.progressbar(
            range(self.iterations), min_value=0, max_value=self.iterations, redirect_stdout=True)

        # new implementation
        start_time = perf_counter()
        if self.parallel_mode and cpu_count() > 1:
            num_procs = max(min(cpu_count() - 2, self.iterations), 1)
            logging.debug(f"Running benchmark in parallel on {num_procs} processors")
            manager = Manager()
            invalid_flag = manager.Value('i', int(False))
            MNE_values = manager.list()
            runtimes = manager.list()
            warnings_dicts = manager.list()
            with get_context('spawn').Pool(num_procs) as pool:    # spawn alternative is forkserver, creates a reusable server
                args = func, args, self.params, invalid_flag
                MNE_values, runtimes, warnings_dicts = zip(*pool.starmap(run_kernel_and_observers, zip(iterator, repeat(args))))
                MNE_values, runtimes, warnings_dicts = list(MNE_values), list(runtimes), list(warnings_dicts)
            result["strategy_time"] = np.mean(runtimes)
            warning_dict = warnings_dicts[0]
            for key in warning_dict.keys():
                warning_dict[key] = np.mean(list(warnings_dict[key] for warnings_dict in warnings_dicts))
            result["warnings"] = warning_dict
        else:
            raise NotImplementedError("Sequential mode has not been implemented yet")

        benchmark_time = perf_counter() - start_time
        self.benchmark_times.append(benchmark_time)
        print(f"Time taken: {round(benchmark_time, 3)} seconds, mean: {round(np.mean(self.benchmark_times), 3)}")

        grandmean, times = get_grandmedian_and_times(MNE_values, invalid_value, min_valid_iterations)
        result["times"] = times
        result["time"] = grandmean
        print(f"Grandmean over kernels: {grandmean}, mean MNE per iteration: {np.mean(times)}, std MNE per iteration: {np.std(times)}")
        return result

        start_time = perf_counter()
        if self.parallel_mode:
            num_procs = max(cpu_count() - 1, 1)
            logging.debug(f"Running benchmark in parallel on {num_procs} processors")
            manager = Manager()
            MRE_values = manager.list()
            runtimes = manager.list()
            with get_context('spawn').Pool(num_procs) as pool:    # spawn alternative is forkserver, creates a reusable server
                args = func, args, self.params
                MRE_values, runtimes = zip(*pool.starmap(run_kernel_and_observers, zip(iterator, repeat(args))))
                MRE_values, runtimes = list(MRE_values), list(runtimes)
                print(MRE_values)
            result["times"] = values
            result["strategy_time"] = np.mean(runtimes)
            np_results = np.array(values)
        else:
            # sequential implementation
            np_results = np.array([])
            for iter in iterator:
                for obs in self.observers:
                    obs.before_start()
                value = self.run_kernel(func, args)
                for obs in self.observers:
                    obs.after_finish()

                if value < 0.0:
                    raise ValueError("Invalid benchmark result")

                result["times"].append(value)
                np_results = np.append(np_results, value)
                if value >= invalid_value and iter >= min_valid_iterations and len(np_results[np_results < invalid_value]) < min_valid_iterations:
                    break

            # fill up the remaining iters with invalid in case of a break
            result["times"] += [invalid_value] * (self.iterations - len(result["times"]))

            # finish by instrumenting the results with the observers
            for obs in self.observers:
                result.update(obs.get_results())

        benchmark_time = perf_counter() - start_time
        self.benchmark_times.append(benchmark_time)
        print(f"Time taken: {round(benchmark_time, 3)} seconds, mean: {round(np.mean(self.benchmark_times), 3)}")

        # calculate the mean of the means of the Mean Relative Error over the valid results
        valid_results = np_results[np_results < invalid_value]
        mean_mean_MRE = np.mean(valid_results) if len(valid_results) > 0 else np.nan

        # write the 'time' to the results and return
        if np.isnan(mean_mean_MRE) or len(valid_results) < min_valid_iterations:
            mean_mean_MRE = invalid_value
        result["time"] = mean_mean_MRE
        return result

    def run_kernel(self, func, args):
        """runs the kernel once, returns whatever the kernel returns

        :param func: A C function compiled for this specific configuration
        :type func: ctypes._FuncPtr

        :param args: A list of arguments to the function, order should match the
            order in the code. The list should be prepared using
            ready_argument_list().
        :type args: list(Argument)

        :param threads: Ignored, but left as argument for now to have the same
            interface as CudaFunctions and OpenCLFunctions.
        :type threads: any

        :param grid: Ignored, but left as argument for now to have the same
            interface as CudaFunctions and OpenCLFunctions.
        :type grid: any

        :returns: A robust average of values returned by the C function.
        :rtype: float
        """
        logging.debug("run_kernel")
        logging.debug("arguments=" + str([str(arg) for arg in args]))

        time = func(*args, **self.params)

        return time

    units = {}


def run_kernel_and_observers(iter, args) -> Tuple[list, float, dict]:
    """ Function to run a kernel directly for parallel processing. Must be outside the class to avoid pickling issues due to large scope. """
    PID = getpid()
    print(f"Iter {iter+1}, PID {PID}", flush=True)
    func, funcargs, params, invalid_flag = args
    logging.debug(f"run_kernel as subprocess {iter} (PID {PID})")
    logging.debug("arguments=" + str([str(arg) for arg in funcargs]))

    # run the kernel
    starttime = perf_counter()
    # cProfile.runctx('func(invalid_flag, *funcargs, **params)', globals(), locals(), 'profile-%s.out' % str(iter + 1))
    # values, warning_dict = None, None
    values, warning_dict = func(invalid_flag, *funcargs, **params)
    runtime = perf_counter() - starttime
    return values, runtime, warning_dict


def run_kernel_as_subprocess(iter, args):
    """ Function to run a kernel as a subprocess for parallel processing. Must be outside the class to avoid pickling issues due to large scope. Significantly slower than run_kernel, but guaranteed to be a different process. Observers are not implemented."""
    func, args, params = args
    PID = getpid()
    # print(f"Iter {iter}, PID {PID}", flush=True)
    logging.debug(f"run_kernel as subprocess {iter} (PID {PID})")
    logging.debug("arguments=" + str([str(arg) for arg in args]))

    def make_kwargstrings(**kwargs) -> list:
        return list(f"{key}={value}" for key, value in kwargs.items())

    # Subprocess
    args += make_kwargstrings(**params)
    proc = subprocess.run([sys.executable or 'python', str(func.__name__ + '.py')] + args, shell=False, capture_output=True)
    stderr = f"subprocess {iter} with PID {PID} errors: {proc.stderr.decode('utf-8')}" if len(proc.stderr.decode('utf-8')) > 0 else ""
    stdout = f"subprocess {iter} with PID {PID} output: {proc.stdout.decode('utf-8')}" if len(proc.stdout.decode('utf-8')) > 0 else ""

    if stderr != "":
        logging.debug(stderr)
        print(stderr)
    if stdout != "":
        logging.debug(stdout)
        # print(stdout)

    time = float(stdout.split("result_value=")[1])
    return time


def get_grandmedian_and_times(MNE_values, invalid_value, min_valid_iterations=1):
    """ Get the grandmean (mean of median MNE per kernel) and mean MNE per iteration """
    MNE_values = np.array(MNE_values)
    median_MNEs = np.array([])
    valid_MNE_times = list()
    # get the mean MNE per kernel
    for i in range(len(MNE_values[0])):
        MNE_kernel_values = MNE_values[:, i]
        valid_MNE_mask = (MNE_kernel_values < invalid_value) & (MNE_kernel_values >= 0)
        valid_MNE_kernel_values = MNE_kernel_values[valid_MNE_mask]
        if len(valid_MNE_kernel_values) >= min_valid_iterations:
            # # filter outliers by keeping only values that are within two times the Median Absolute Deviation
            # AD = np.abs(valid_MNE_kernel_values - np.median(valid_MNE_kernel_values))
            # MAD = np.median(AD)
            # selected_MNE_kernel_values = valid_MNE_kernel_values[AD < MAD * 3]
            # print(f"Removed {len(valid_MNE_kernel_values) - len(selected_MNE_kernel_values)}")
            # median_MNEs = np.append(median_MNEs, np.median(selected_MNE_kernel_values))
            # median_MNEs = np.append(median_MNEs, np.mean(valid_MNE_kernel_values))

            # filter outliers by keeping only values that are within three times the Median Absolute Deviation
            AD = np.abs(valid_MNE_kernel_values - np.median(valid_MNE_kernel_values))
            MAD = np.median(AD)
            MAD_score = AD / MAD if MAD else 0.0
            selected_MNE_kernel_values = valid_MNE_kernel_values[MAD_score < 3]
            median_MNEs = np.append(median_MNEs, np.median(selected_MNE_kernel_values))
        else:
            median_MNEs = np.append(median_MNEs, invalid_value)

    # get the mean MNE per iteration
    for i in range(len(MNE_values)):
        MNE_iteration_values = MNE_values[i]
        valid_MNE_mask = (MNE_iteration_values < invalid_value) & (MNE_iteration_values >= 0)
        valid_MNE_iteration_values = MNE_iteration_values[valid_MNE_mask]
        if len(valid_MNE_iteration_values) > 0:
            valid_MNE_times.append(np.mean(valid_MNE_iteration_values))
        else:
            valid_MNE_times.append(invalid_value)

    # get the grandmean by taking the mean over the median MNE per iteration, invalid if one of the kernels is invalid
    print(median_MNEs)
    grandmean_MNE = np.mean(median_MNEs)
    if np.isnan(grandmean_MNE) or len(median_MNEs[median_MNEs >= invalid_value]) > 0:
        grandmean_MNE = invalid_value
    return grandmean_MNE, valid_MNE_times
