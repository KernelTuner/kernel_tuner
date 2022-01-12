""" This module contains the functionality for running and compiling C functions """

from collections import namedtuple
import platform
import logging
import importlib.util

import numpy
import numpy.ctypeslib

from kernel_tuner.util import get_temp_filename, delete_temp_file, write_file

# This represents an individual kernel argument.
# It contains a numpy object (ndarray or number) and a ctypes object with a copy
# of the argument data. For an ndarray, the ctypes object is a wrapper for the ndarray's data.
Argument = namedtuple("Argument", ["numpy", "ctypes"])


class PythonFunctions(object):
    """Class that groups the code for running and compiling C functions"""

    def __init__(self, iterations=7):
        """instantiate PythonFunctions object used for interacting with Python code

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int
        """
        self.iterations = iterations
        self.max_threads = 1024

        #environment info
        env = dict()
        env["iterations"] = self.iterations
        self.env = env
        self.name = platform.processor()

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
        result = dict()
        result["times"] = []
        for _ in range(self.iterations):
            value = self.run_kernel(func, args, threads, grid)

            #I would like to replace the following with actually capturing
            #stderr and detecting the error directly in Python, it proved
            #however that capturing stderr for non-Python functions from Python
            #is a rather difficult thing to do
            #
            #The current, less than ideal, scheme uses the convention that a
            #negative time indicates a 'too many resources requested for launch'
            #which Kernel Tuner can silently ignore
            if value < 0.0:
                raise Exception("too many resources requested for launch")

            result["times"].append(value)
        result["time"] = numpy.mean(result["times"])
        return result

    def run_kernel(self, func, args, threads, grid):
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

        time = func(**self.params)

        return time

    units = {}
