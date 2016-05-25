""" This module contains the functionality for running and compiling C functions """

import numpy
import ctypes as C
import subprocess
import os
import errno

import numpy.ctypeslib

class CFunctions(object):
    """Class that groups the code for running and compiling C functions"""

    def __init__(self, iterations=7):
        """instantiate CFunctions object used for interacting with C code

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int
        """
        self.ITERATIONS = iterations
        self.max_threads = 1024

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the C function

        :param arguments: List of arguments to be passed to the C function.
            The order should match the argument list on the C function.
            Allowed values are numpy.ndarray, and/or numpy.int32, numpy.float32, and so on.
        :type arguments: list(numpy objects)

        :returns: A list of arguments that can be passed to the C function.
        :rtype: list()
        """
        ctype_args = []
        for arg in arguments:
            if isinstance(arg, numpy.ndarray):
                if arg.dtype == 'float32':
                    ctype_args.append(arg.ctypes.data_as(C.POINTER(C.c_float)))
                elif arg.dtype == 'float64':
                    ctype_args.append(arg.ctypes.data_as(C.POINTER(C.c_double)))
                elif arg.dtype == 'int32':
                    ctype_args.append(arg.ctypes.data_as(C.POINTER(C.c_int)))
                else:
                    raise TypeError("unknown dtype for ndarray")
            elif numpy.isscalar(arg):
                if hasattr(arg, 'dtype'):
                    if str(arg.dtype).startswith('int'):
                        ctype_args.append(int(arg))
                    elif str(arg.dtype).startswith('float'):
                        ctype_args.append(float(arg))
                    else:
                        raise TypeError("Argument is scalar with a dtype, but does not start with int or float")
                else:
                    ctype_args.append(arg)
            else:
                raise TypeError("Argument is not a numpy.ndarray and is not a scalar %s" % type(arg))

        return ctype_args


    def compile(self, kernel_name, kernel_string):
        """call the C compiler to compile the kernel, return the function

        :param kernel_name: The name of the kernel to be compiled, used to lookup the
            function after compilation.
        :type kernel_name: string

        :param kernel_string: The C code that contains the function `kernel_name`
        :type kernel_string: string

        :returns: An ctypes function that can be called directly.
        :rtype: ctypes._FuncPtr
        """
        random_large_int = numpy.random.randint(low=1000000, high=1000000000)
        filename = 'temp_' + str(random_large_int)
        source_file = filename+".cu"

        kernel_string = "extern \"C\" {\n" + kernel_string + "\n}"

        try:
            with open(source_file, 'w') as f:
                f.write(kernel_string)

            subprocess.check_call(["nvcc", "-c", source_file, "-Xcompiler=-fPIC", "-o", filename+".o"])
            subprocess.check_call(["nvcc", filename+".o", "-shared", "-o", filename+".so"])

            self.lib = numpy.ctypeslib.load_library(filename, '.')

        finally:
            _delete_temp_file(source_file)
            _delete_temp_file(filename+".o")
            _delete_temp_file(filename+".so")

        func = getattr(self.lib, kernel_name)
        func.restype = C.c_float

        return func

    def benchmark(self, func, c_args, threads, grid):
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

        :param c_args: A list of arguments to the function, order should match the
            order in the code. The list should be prepared using
            ready_argument_list().
        :type c_args: list()

        :param threads: Ignored, but left as argument for now to have the same
            interface as CudaFunctions and OpenCLFunctions.
        :type threads: any

        :param grid: Ignored, but left as argument for now to have the same
            interface as CudaFunctions and OpenCLFunctions.
        :type grid: any

        :returns: A robust average of values returned by the C function.
        :rtype: float
        """
        results = []
        for _ in range(self.ITERATIONS):
            value = func(*c_args)
            results.append(value)
        results = sorted(results)
        return numpy.mean(results[1:-1])




def _delete_temp_file(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise e
