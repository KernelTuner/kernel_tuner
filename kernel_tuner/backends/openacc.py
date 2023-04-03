""" This module contains the functionality for running and compiling OpenACC sections """

import subprocess
import platform

from kernel_tuner.backends.backend import CompilerBackend
from kernel_tuner.observers.c import CRuntimeObserver


class OpenACCFunctions(CompilerBackend):
    """Class that groups the code for running and compiling OpenaCC functions in C++."""

    def __init__(self, iterations=7, compiler_options=None, compiler=None):
        self.iterations = iterations
        # if no compiler is specified, use nvc++ by default
        self.compiler = compiler or "nvc++"
        self.observers = [CRuntimeObserver(self)]

        cc_version = str(subprocess.check_output([self.compiler, "--version"]))
        cc_version = cc_version.splitlines()[0].split(" ")[-1]

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
        raise NotImplementedError("OpenACC backend does not support this feature")

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
