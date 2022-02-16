""" This module contains the functionality for running and compiling C functions """

from collections import namedtuple
import subprocess
import platform
import errno
import re
import logging
import ctypes as C
import _ctypes

import numpy
import numpy.ctypeslib

from kernel_tuner.util import get_temp_filename, delete_temp_file, write_file

dtype_map = {"int8": C.c_int8,
             "int16": C.c_int16,
             "int32": C.c_int32,
             "int64": C.c_int64,
             "uint8": C.c_uint8,
             "uint16": C.c_uint16,
             "uint32": C.c_uint32,
             "uint64": C.c_uint64,
             "float32": C.c_float,
             "float64": C.c_double}

# This represents an individual kernel argument.
# It contains a numpy object (ndarray or number) and a ctypes object with a copy
# of the argument data. For an ndarray, the ctypes object is a wrapper for the ndarray's data.
Argument = namedtuple("Argument", ["numpy", "ctypes"])


class CFunctions(object):
    """Class that groups the code for running and compiling C functions"""

    def __init__(self, iterations=7, compiler_options=None, compiler=None):
        """instantiate CFunctions object used for interacting with C code

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int
        """
        self.iterations = iterations
        self.max_threads = 1024
        self.compiler_options = compiler_options
        self.compiler = compiler or "g++"  # use gcc by default
        self.lib = None
        self.using_openmp = False

        try:
            cc_version = str(subprocess.check_output([self.compiler, "--version"]))
            cc_version = cc_version.splitlines()[0].split(" ")[-1]
        except OSError as e:
            raise e

        #check if nvcc is available
        self.nvcc_available = False
        try:
            nvcc_version = str(subprocess.check_output(["nvcc", "--version"]))
            nvcc_version = nvcc_version.splitlines()[-1].split(" ")[-1]
            self.nvcc_available = True
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise e

        #environment info
        env = dict()
        env["CC Version"] = cc_version
        if self.nvcc_available:
            env["NVCC Version"] = nvcc_version
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        self.env = env
        self.name = platform.processor()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the C function

        :param arguments: List of arguments to be passed to the C function.
            The order should match the argument list on the C function.
            Allowed values are numpy.ndarray, and/or numpy.int32, numpy.float32, and so on.
        :type arguments: list(numpy objects)

        :returns: A list of arguments that can be passed to the C function.
        :rtype: list(Argument)
        """
        ctype_args = [ None for _ in arguments]

        for i, arg in enumerate(arguments):
            if not isinstance(arg, (numpy.ndarray, numpy.number)):
                raise TypeError("Argument is not numpy ndarray or numpy scalar %s" % type(arg))
            dtype_str = str(arg.dtype)
            #data = arg.copy()
            if isinstance(arg, numpy.ndarray):
                if dtype_str in dtype_map.keys():
                    # In numpy <= 1.15, ndarray.ctypes.data_as does not itself keep a reference
                    # to its underlying array, so we need to store a reference to arg.copy()
                    # in the Argument object manually to avoid it being deleted.
                    # (This changed in numpy > 1.15.)
                    #data_ctypes = data.ctypes.data_as(C.POINTER(dtype_map[dtype_str]))
                    data_ctypes = arg.ctypes.data_as(C.POINTER(dtype_map[dtype_str]))
                else:
                    raise TypeError("unknown dtype for ndarray")
            elif isinstance(arg, numpy.generic):
                data_ctypes = dtype_map[dtype_str](arg)
            #ctype_args[i] = Argument(numpy=data, ctypes=data_ctypes)
            ctype_args[i] = Argument(numpy=arg, ctypes=data_ctypes)
        return ctype_args

    def compile(self, kernel_instance):
        """call the C compiler to compile the kernel, return the function

        :param kernel_instance: An object representing the specific instance of the tunable kernel
            in the parameter space.
        :type kernel_instance: kernel_tuner.core.KernelInstance

        :returns: An ctypes function that can be called directly.
        :rtype: ctypes._FuncPtr
        """
        logging.debug('compiling ' + kernel_instance.name)

        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name

        if self.lib != None:
            self.cleanup_lib()

        compiler_options = ["-fPIC"]

        #detect openmp
        if "#include <omp.h>" in kernel_string or "use omp_lib" in kernel_string:
            logging.debug('set using_openmp to true')
            self.using_openmp = True
            if self.compiler == "pgfortran":
                compiler_options.append("-mp")
            else:
                if "#pragma acc" in kernel_string or "!$acc" in kernel_string:
                    compiler_options.append("-fopenacc")
                else:
                    compiler_options.append("-fopenmp")

        #if filename is known, use that one
        suffix = kernel_instance.kernel_source.get_user_suffix()

        #if code contains device code, suffix .cu is required
        device_code_signals = ["__global", "__syncthreads()", "threadIdx"]
        if any([snippet in kernel_string for snippet in device_code_signals]):
            suffix = ".cu"

        #detect whether to use nvcc as default instead of g++, may overrule an explicitly passed g++
        if (suffix == ".cu") or ("#include <cuda" in kernel_string) or ("cudaMemcpy" in kernel_string):
            if self.compiler == "g++" and self.nvcc_available:
                self.compiler = "nvcc"

        if suffix is None:
            #select right suffix based on compiler
            suffix = ".cc"

            if self.compiler in ["gfortran", "pgfortran", "ftn", "ifort"]:
                suffix = ".F90"

        if self.compiler == "nvcc":
            compiler_options = ["-Xcompiler=" + c for c in compiler_options]

        #this basically checks if we aren't compiling Fortran
        #at the moment any C, C++, or CUDA code is assumed to use extern "C" linkage
        if ".c" in suffix:
            if not "extern \"C\"" in kernel_string:
                kernel_string = "extern \"C\" {\n" + kernel_string + "\n}"

        #copy user specified compiler options to current list
        if self.compiler_options:
            compiler_options += self.compiler_options

        lib_args = []
        if "CL/cl.h" in kernel_string:
            lib_args = ["-lOpenCL"]

        logging.debug('using compiler ' + self.compiler)
        logging.debug('compiler_options ' + " ".join(compiler_options))
        logging.debug('lib_args ' + " ".join(lib_args))

        source_file = get_temp_filename(suffix=suffix)
        filename = ".".join(source_file.split(".")[:-1])

        #detect Fortran modules
        match = re.search(r"\s*module\s+([a-zA-Z_]*)", kernel_string)
        if match:
            if self.compiler == "gfortran":
                kernel_name = "__" + match.group(1) + "_MOD_" + kernel_name
            elif self.compiler in ["ftn", "ifort"]:
                kernel_name = match.group(1) + "_mp_" + kernel_name + "_"
            elif self.compiler == "pgfortran":
                kernel_name = match.group(1) + "_" + kernel_name + "_"
        else:
            #for functions outside of modules
            if self.compiler in ["gfortran", "ftn", "ifort", "pgfortran"]:
                kernel_name = kernel_name + "_"

        try:
            write_file(source_file, kernel_string)

            lib_extension = ".so"
            if platform.system() == "Darwin":
                lib_extension = ".dylib"

            subprocess.check_call([self.compiler, "-c", source_file] + compiler_options + ["-o", filename + ".o"])
            subprocess.check_call([self.compiler, filename + ".o"] + compiler_options + ["-shared", "-o", filename + lib_extension] + lib_args)

            self.lib = numpy.ctypeslib.load_library(filename, '.')
            func = getattr(self.lib, kernel_name)
            func.restype = C.c_float

        finally:
            delete_temp_file(source_file)
            delete_temp_file(filename+".o")
            delete_temp_file(filename+".so")
            delete_temp_file(filename+".dylib")

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
        :type c_args: list(Argument)

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
            value = self.run_kernel(func, c_args, threads, grid)

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

    def run_kernel(self, func, c_args, threads, grid):
        """runs the kernel once, returns whatever the kernel returns

        :param func: A C function compiled for this specific configuration
        :type func: ctypes._FuncPtr

        :param c_args: A list of arguments to the function, order should match the
            order in the code. The list should be prepared using
            ready_argument_list().
        :type c_args: list(Argument)

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
        logging.debug("arguments=" + str([str(arg.ctypes) for arg in c_args]))

        time = func(*[arg.ctypes for arg in c_args])

        return time

    def memset(self, allocation, value, size):
        """set the memory in allocation to the value in value

        :param allocation: An Argument for some memory allocation unit
        :type allocation: Argument

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int
        """
        C.memset(allocation.ctypes, value, size)

    def memcpy_dtoh(self, dest, src):
        """a simple memcpy copying from an Argument to a numpy array

        :param dest: A numpy array to store the data
        :type dest: numpy.ndarray

        :param src: An Argument for some memory allocation
        :type src: Argument
        """
        dest[:] = src.numpy

    def memcpy_htod(self, dest, src):
        """a simple memcpy copying from a numpy array to an Argument

        :param dest: An Argument for some memory allocation
        :type dst: Argument

        :param src: A numpy array containing the source data
        :type src: numpy.ndarray
        """
        dest.numpy[:] = src

    def cleanup_lib(self):
        """ unload the previously loaded shared library """
        if not self.using_openmp:
            #this if statement is necessary because shared libraries that use
            #OpenMP will core dump when unloaded, this is a well-known issue with OpenMP
            logging.debug('unloading shared library')
            _ctypes.dlclose(self.lib._handle)

    units = {}
