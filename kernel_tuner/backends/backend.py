"""This module contains the interface of all kernel_tuner backends"""
from __future__ import print_function

from abc import ABC, abstractmethod


class Backend(ABC):
    """Base class for kernel_tuner backends"""

    @abstractmethod
    def ready_argument_list(self, arguments):
        """This method must implement the allocation of the arguments on device memory."""
        pass

    @abstractmethod
    def compile(self, kernel_instance):
        """This method must implement the compilation of a kernel into a callable function."""
        pass

    @abstractmethod
    def start_event(self):
        """This method must implement the recording of the start of a measurement."""
        pass

    @abstractmethod
    def stop_event(self):
        """This method must implement the recording of the end of a measurement."""
        pass

    @abstractmethod
    def kernel_finished(self):
        """This method must implement a check that returns True if the kernel has finished, False otherwise."""
        pass

    @abstractmethod
    def synchronize(self):
        """This method must implement a barrier that halts execution until device has finished its tasks."""
        pass

    @abstractmethod
    def run_kernel(self, func, gpu_args, threads, grid, stream):
        """This method must implement the execution of the kernel on the device."""
        pass

    @abstractmethod
    def memset(self, allocation, value, size):
        """This method must implement setting the memory to a value on the device."""
        pass

    @abstractmethod
    def memcpy_dtoh(self, dest, src):
        """This method must implement a device to host copy."""
        pass

    @abstractmethod
    def memcpy_htod(self, dest, src):
        """This method must implement a host to device copy."""
        pass


class GPUBackend(Backend):
    """Base class for GPU backends"""

    @abstractmethod
    def __init__(self, device, iterations, compiler_options, observers):
        pass

    @abstractmethod
    def copy_constant_memory_args(self, cmem_args):
        """This method must implement the allocation and copy of constant memory to the GPU."""
        pass

    @abstractmethod
    def copy_shared_memory_args(self, smem_args):
        """This method must implement the dynamic allocation of shared memory on the GPU."""
        pass

    @abstractmethod
    def copy_texture_memory_args(self, texmem_args):
        """This method must implement the allocation and copy of texture memory to the GPU."""
        pass


class CompilerBackend(Backend):
    """Base class for compiler backends"""

    @abstractmethod
    def __init__(self, iterations, compiler_options, compiler):
        pass
