""" This module contains the functionality for running and compiling OpenACC sections """

import subprocess
import platform
import ctypes as C
import _ctypes
import numpy as np

from kernel_tuner.backends.backend import CompilerBackend
from kernel_tuner.observers.c import CRuntimeObserver
from kernel_tuner.util import get_temp_filename, write_file, delete_temp_file


class OpenACCFunctions(CompilerBackend):
    """Class that groups the code for running and compiling OpenaCC functions in C++."""

    def __init__(self, iterations=7, compiler_options=None, compiler="nvc++"):
        self.iterations = iterations
        # if no compiler is specified, use nvc++ by default
        self.compiler = compiler
        self.lib = None
        self.observers = [CRuntimeObserver(self)]

        cc_version = str(subprocess.check_output([self.compiler, "--version"]))
        cc_version = cc_version.splitlines()[0].split(" ")[1]

        # environment info
        env = dict()
        env["CC Version"] = cc_version
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        self.env = env
        self.name = platform.processor()

    def ready_argument_list(self, arguments):
        """This method must implement the allocation of the arguments on device memory."""
        raise NotImplementedError("OpenACC backend does not support this feature")

    def compile(self, kernel_instance):
        """This method must implement the compilation of a kernel into a callable function."""
        if self.lib is not None:
            self.cleanup_lib()
        compiler_options = ["-fPIC -fast -acc=gpu"]
        if self.compiler_options:
            compiler_options += self.compiler_options
        source_file = get_temp_filename(suffix=".cpp")
        filename = ".".join(source_file.split(".")[:-1])
        try:
            write_file(source_file, kernel_instance.kernel_string)

            lib_extension = ".so"
            if platform.system() == "Darwin":
                lib_extension = ".dylib"

            subprocess.check_call(
                [self.compiler, "-c", source_file]
                + compiler_options
                + ["-o", filename + ".o"]
            )
            subprocess.check_call(
                [self.compiler, filename + ".o"]
                + compiler_options
                + ["-shared", "-o", filename + lib_extension]
            )

            self.lib = np.ctypeslib.load_library(filename, ".")
            func = getattr(self.lib, kernel_instance.kernel_name)
            func.restype = C.c_float

        finally:
            delete_temp_file(source_file)
            delete_temp_file(filename + ".o")
            delete_temp_file(filename + ".so")
            delete_temp_file(filename + ".dylib")

        return func

    def start_event(self):
        """This method must implement the recording of the start of a measurement."""
        raise NotImplementedError("OpenACC backend does not support this feature")

    def stop_event(self):
        """This method must implement the recording of the end of a measurement."""
        raise NotImplementedError("OpenACC backend does not support this feature")

    def kernel_finished(self):
        """This method must implement a check that returns True if the kernel has finished, False otherwise."""
        raise NotImplementedError("OpenACC backend does not support this feature")

    def synchronize(self):
        """This method must implement a barrier that halts execution until device has finished its tasks."""
        raise NotImplementedError("OpenACC backend does not support this feature")

    def run_kernel(self, func, gpu_args, threads, grid, stream):
        """This method must implement the execution of the kernel on the device."""
        raise NotImplementedError("OpenACC backend does not support this feature")

    def memset(self, allocation, value, size):
        """This method must implement setting the memory to a value on the device."""
        raise NotImplementedError("OpenACC backend does not support this feature")

    def memcpy_dtoh(self, dest, src):
        """This method must implement a device to host copy."""
        raise NotImplementedError("OpenACC backend does not support this feature")

    def memcpy_htod(self, dest, src):
        """This method must implement a host to device copy."""
        raise NotImplementedError("OpenACC backend does not support this feature")

    def cleanup_lib(self):
        """Unload the previously loaded shared library"""
        _ctypes.dlclose(self.lib._handle)
