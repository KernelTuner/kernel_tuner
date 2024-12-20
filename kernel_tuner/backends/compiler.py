""" This module contains the functionality for running and compiling C functions """

from collections import namedtuple
import subprocess
import platform
import errno
import re
import logging
import ctypes as C
import _ctypes

import numpy as np
import numpy.ctypeslib

from kernel_tuner.backends.backend import CompilerBackend
from kernel_tuner.observers.compiler import CompilerRuntimeObserver
from kernel_tuner.util import (
    get_temp_filename,
    delete_temp_file,
    write_file,
    SkippableFailure,
)

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from hip import hip
except ImportError:
    hip = None

try:
    from hip._util.types import DeviceArray
except ImportError:
    Pointer = Exception # using Exception here as a type that will never be among kernel arguments
    DeviceArray = Exception


def is_cupy_array(array):
    """Check if something is a cupy array.

    :param array: A Python object.
    :type array: typing.Any

    :returns: True if cupy can be imported and the object is a cupy.ndarray.
    :rtype: bool
    """
    return cp is not None and isinstance(array, cp.ndarray)


def get_array_module(*args):
    """Return the array module for arguments.

    This function is used to implement CPU/GPU generic code. If the cupy module can be imported
    and at least one of the arguments is a cupy.ndarray object, the cupy module is returned.

    :param args: Values to determine whether NumPy or CuPy should be used.
    :type args: numpy.ndarray or cupy.ndarray

    :returns: cupy or numpy is returned based on the types of the arguments.
    :rtype: types.ModuleType
    """
    return np if cp is None else cp.get_array_module(*args)


dtype_map = {
    "int8": C.c_int8,
    "int16": C.c_int16,
    "int32": C.c_int32,
    "int64": C.c_int64,
    "uint8": C.c_uint8,
    "uint16": C.c_uint16,
    "uint32": C.c_uint32,
    "uint64": C.c_uint64,
    "float32": C.c_float,
    "float64": C.c_double,
}

# This represents an individual kernel argument.
# It contains a numpy object (ndarray or number) and a ctypes object with a copy
# of the argument data. For an ndarray, the ctypes object is a wrapper for the ndarray's data.
Argument = namedtuple("Argument", ["numpy", "ctypes"])


class CompilerFunctions(CompilerBackend):
    """Class that groups the code for running and compiling C functions"""

    def __init__(self, iterations=7, compiler_options=None, compiler=None, observers=None):
        """instantiate CFunctions object used for interacting with C code

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int
        """
        self.observers = observers or []
        self.observers.append(CompilerRuntimeObserver(self))

        self.iterations = iterations
        self.max_threads = 1024
        self.compiler_options = compiler_options
        # if no compiler is specified, use g++ by default
        self.compiler = compiler or "g++"
        self.lib = None
        self.using_openmp = False
        self.using_openacc = False
        self.observers = [CompilerRuntimeObserver(self)]
        self.last_result = None

        if self.compiler == "g++":
            try:
                cc_version = str(subprocess.check_output([self.compiler, "--version"]))
                cc_version = cc_version.split("\\n")[0].split(" ")[2]
            except OSError as e:
                raise e
        elif self.compiler in ["nvc", "nvc++", "nvfortran"]:
            try:
                cc_version = str(subprocess.check_output([self.compiler, "--version"]))
                cc_version = cc_version.split(" ")[1]
            except OSError as e:
                raise e
        else:
            cc_version = None

        # check if nvcc is available
        self.nvcc_available = False
        try:
            nvcc_version = str(subprocess.check_output(["nvcc", "--version"]))
            nvcc_version = nvcc_version.splitlines()[-1].split(" ")[-1]
            self.nvcc_available = True
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise e

        # environment info
        env = dict()
        env["CC Version"] = cc_version
        if self.nvcc_available:
            env["NVCC Version"] = nvcc_version
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        self.env = env
        self.name = platform.processor()

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the C function

        :param arguments: List of arguments to be passed to the C function.
            The order should match the argument list on the C function.
            Allowed values are np.ndarray, cupy.ndarray, and/or np.int32, np.float32, and so on.
        :type arguments: list(numpy or cupy objects)

        :returns: A list of arguments that can be passed to the C function.
        :rtype: list(Argument)
        """
        ctype_args = [None for _ in arguments]

        for i, arg in enumerate(arguments):
            if not (isinstance(arg, (np.ndarray, np.number, DeviceArray)) or is_cupy_array(arg)):
                raise TypeError(f"Argument is not numpy or cupy ndarray or numpy scalar or HIP Python DeviceArray but a {type(arg)}")
            dtype_str = arg.typestr if isinstance(arg, DeviceArray) else str(arg.dtype)
            if isinstance(arg, np.ndarray):
                if dtype_str in dtype_map.keys():
                    # In numpy <= 1.15, ndarray.ctypes.data_as does not itself keep a reference
                    # to its underlying array, so we need to store a reference to arg.copy()
                    # in the Argument object manually to avoid it being deleted.
                    # (This changed in numpy > 1.15.)
                    # data_ctypes = data.ctypes.data_as(C.POINTER(dtype_map[dtype_str]))
                    data_ctypes = arg.ctypes.data_as(C.POINTER(dtype_map[dtype_str]))
                    numpy_arg = arg
                else:
                    raise TypeError("unknown dtype for ndarray")
            elif isinstance(arg, np.generic):
                data_ctypes = dtype_map[dtype_str](arg)
                numpy_arg = arg
            elif is_cupy_array(arg):
                data_ctypes = C.c_void_p(arg.data.ptr)
                numpy_arg = arg
            elif isinstance(arg, DeviceArray):
                data_ctypes = arg.as_c_void_p()
                numpy_arg = None

            ctype_args[i] = Argument(numpy=numpy_arg, ctypes=data_ctypes)
        return ctype_args

    def compile(self, kernel_instance):
        """call the C compiler to compile the kernel, return the function

        :param kernel_instance: An object representing the specific instance of the tunable kernel
            in the parameter space.
        :type kernel_instance: kernel_tuner.core.KernelInstance

        :returns: An ctypes function that can be called directly.
        :rtype: ctypes._FuncPtr
        """
        logging.debug("compiling " + kernel_instance.name)

        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name

        if self.lib is not None:
            self.cleanup_lib()

        compiler_options = ["-fPIC"]

        # detect openmp
        if "#include <omp.h>" in kernel_string or "use omp_lib" in kernel_string:
            logging.debug("set using_openmp to true")
            self.using_openmp = True
            if self.compiler in ["nvc", "nvc++", "nvfortran"]:
                compiler_options.append("-mp")
            else:
                compiler_options.append("-fopenmp")

        # detect openacc
        if "#pragma acc" in kernel_string or "!$acc" in kernel_string:
            self.using_openacc = True

        # if filename is known, use that one
        suffix = kernel_instance.kernel_source.get_user_suffix()

        # if code contains device code, suffix .cu is required
        device_code_signals = ["__global", "__syncthreads()", "threadIdx"]
        if any([snippet in kernel_string for snippet in device_code_signals]):
            suffix = ".cu"

        # detect whether to use nvcc as default instead of g++, may overrule an explicitly passed g++
        if (
            ((suffix == ".cu") or ("#include <cuda" in kernel_string) or ("cudaMemcpy" in kernel_string))
            and self.compiler == "g++"
            and self.nvcc_available
        ):
            self.compiler = "nvcc"

        if suffix is None:
            # select right suffix based on compiler
            suffix = ".cc"

            if self.compiler in ["gfortran", "nvfortran", "ftn", "ifort"]:
                suffix = ".F90"

        if self.compiler == "nvcc":
            compiler_options = ["-Xcompiler=" + c for c in compiler_options]

        # this basically checks if we aren't compiling Fortran
        # at the moment any C, C++, or CUDA code is assumed to use extern "C" linkage
        if ".c" in suffix and 'extern "C"' not in kernel_string:
            kernel_string = 'extern "C" {\n' + kernel_string + "\n}"

        # copy user specified compiler options to current list
        if self.compiler_options:
            compiler_options += self.compiler_options

        lib_args = []
        if "CL/cl.h" in kernel_string:
            lib_args = ["-lOpenCL"]

        logging.debug("using compiler " + self.compiler)
        logging.debug("compiler_options " + " ".join(compiler_options))
        logging.debug("lib_args " + " ".join(lib_args))

        source_file = get_temp_filename(suffix=suffix)
        filename = ".".join(source_file.split(".")[:-1])

        # detect Fortran modules
        match = re.search(r"\s*module\s+([a-zA-Z_]*)", kernel_string)
        if match:
            if self.compiler == "gfortran":
                kernel_name = "__" + match.group(1) + "_MOD_" + kernel_name
            elif self.compiler in ["ftn", "ifort"]:
                kernel_name = match.group(1) + "_mp_" + kernel_name + "_"
            elif self.compiler == "nvfortran":
                kernel_name = match.group(1) + "_" + kernel_name + "_"
        else:
            # for functions outside of modules
            if self.compiler in ["gfortran", "ftn", "ifort", "nvfortran"]:
                kernel_name = kernel_name + "_"

        try:
            write_file(source_file, kernel_string)

            lib_extension = ".so"
            if platform.system() == "Darwin":
                lib_extension = ".dylib"

            subprocess.run(
                [self.compiler, "-c", source_file] + compiler_options + ["-o", filename + ".o"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            subprocess.run(
                [self.compiler, filename + ".o"]
                + compiler_options
                + ["-shared", "-o", filename + lib_extension]
                + lib_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            self.lib = np.ctypeslib.load_library(filename, ".")
            func = getattr(self.lib, kernel_name)
            func.restype = C.c_float

        finally:
            delete_temp_file(source_file)
            delete_temp_file(filename + ".o")
            delete_temp_file(filename + ".so")
            delete_temp_file(filename + ".dylib")

        return func

    def start_event(self):
        """Records the event that marks the start of a measurement

        C backend does not use events"""
        pass

    def stop_event(self):
        """Records the event that marks the end of a measurement

        C backend does not use events"""
        pass

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise

        C backend does not support asynchronous launches"""
        return True

    def synchronize(self):
        """Halts execution until device has finished its tasks

        C backend does not support asynchronous launches"""
        pass

    def run_kernel(self, func, c_args, threads, grid, stream=None):
        """runs the kernel once, returns whatever the kernel returns

        :param func: A C function compiled for this specific configuration
        :type func: ctypes._FuncPtr

        :param c_args: A list of arguments to the function, order should match the
            order in the code. The list should be prepared using
            ready_argument_list().
        :type c_args: list(Argument)

        :param threads: Ignored, but left as argument for now to have the same
            interface as Backend.
        :type threads: any

        :param grid: Ignored, but left as argument for now to have the same
            interface as Backend.
        :type grid: any

        :param stream: Ignored, but left as argument for now to have the same
            interface as Backend.
        :type grid: any

        :returns: A robust average of values returned by the C function.
        :rtype: float
        """
        logging.debug("run_kernel")
        logging.debug("arguments=" + str([str(arg.ctypes) for arg in c_args]))

        time = func(*[arg.ctypes for arg in c_args])
        self.last_result = time

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
        if is_cupy_array(allocation.numpy):
            cp.cuda.runtime.memset(allocation.numpy.data.ptr, value, size)
        else:
            C.memset(allocation.ctypes, value, size)

    def memcpy_dtoh(self, dest, src):
        """a simple memcpy copying from an Argument to a numpy array

        :param dest: A numpy or cupy array to store the data
        :type dest: np.ndarray or cupy.ndarray

        :param src: An Argument for some memory allocation
        :type src: Argument
        """
        # If src.numpy is None, it means we're dealing with a HIP Python DeviceArray
        if src.numpy is None:
            # Skip memory copies for HIP Python DeviceArray
            # This is because DeviceArray manages its own memory and donesn't need
            # explicit copies like numpy arrays do
            return
        if isinstance(dest, np.ndarray) and is_cupy_array(src.numpy):
            # Implicit conversion to a NumPy array is not allowed.
            value = src.numpy.get()
        else:
            value = src.numpy
        xp = get_array_module(dest)
        dest[:] = xp.asarray(value)

    def memcpy_htod(self, dest, src):
        """a simple memcpy copying from a numpy array to an Argument

        :param dest: An Argument for some memory allocation
        :type dest: Argument

        :param src: A numpy or cupy array containing the source data
        :type src: np.ndarray or cupy.ndarray
        """
        # If src.numpy is None, it means we're dealing with a HIP Python DeviceArray
        if dest.numpy is None:
            # Skip memory copies for HIP Python DeviceArray
            # This is because DeviceArray manages its own memory and donesn't need
            # explicit copies like numpy arrays do
            return
        if isinstance(dest.numpy, np.ndarray) and is_cupy_array(src):
            # Implicit conversion to a NumPy array is not allowed.
            value = src.get()
        else:
            value = src
        xp = get_array_module(dest.numpy)
        dest.numpy[:] = xp.asarray(value)

    def cleanup_lib(self):
        """unload the previously loaded shared library"""
        if self.lib is None:
            return
        
        if not self.using_openmp and not self.using_openacc:
            # this if statement is necessary because shared libraries that use
            # OpenMP will core dump when unloaded, this is a well-known issue with OpenMP
            logging.debug("unloading shared library")
            try:
                _ctypes.dlclose(self.lib._handle)
            finally:
                self.lib = None

    units = {}
