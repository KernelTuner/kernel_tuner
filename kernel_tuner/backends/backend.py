"""This module contains the interface of all kernel_tuner backends."""
from __future__ import print_function

from abc import ABC, abstractmethod

import numpy as np


def get_device_array(arg):
    """Return (device pointer, size in bytes) if arg implements the CUDA Array Interface, None otherwise.

    This covers for example PyTorch CUDA tensors, CuPy arrays, and Numba device arrays,
    without having to import any of these libraries.
    """
    # CPU tensors raise AttributeError on this property, which getattr turns into None
    cai = getattr(arg, "__cuda_array_interface__", None)
    if cai is None:
        return None
    shape = tuple(cai["shape"])
    itemsize = np.dtype(cai["typestr"]).itemsize
    strides = cai.get("strides")
    if strides is not None:
        contiguous = tuple(int(np.prod(shape[i + 1 :])) * itemsize for i in range(len(shape)))
        if tuple(strides) != contiguous:
            raise ValueError("Device arrays passed as kernel arguments must be C-contiguous")
    return cai["data"][0], int(np.prod(shape)) * itemsize


def is_host_array(arg):
    """Return True if arg is a numpy array or a host array that converts to one, such as a CPU PyTorch tensor."""
    if isinstance(arg, np.ndarray):
        return True
    return hasattr(arg, "__array__") and get_device_array(arg) is None and not np.isscalar(arg)


class Backend(ABC):
    """Base class for kernel_tuner backends."""

    @abstractmethod
    def ready_argument_list(self, arguments):
        """This method must implement the allocation of the arguments on device memory."""
        return arguments

    @abstractmethod
    def compile(self, kernel_instance):
        """This method must implement the compilation of a kernel into a callable function."""
        pass

    def build(self, kernel_instance):
        """Compile a kernel without loading it onto the device, the result is passed to load().

        Kernel Tuner can call build() for multiple kernels in parallel threads, so implementations must be
        thread-safe and must not modify the state of the backend. Backends that do not implement build()
        compile the kernel in load() instead.
        """
        return None

    def load(self, kernel_instance, build_result):
        """Load a kernel compiled by build() onto the device and return a callable function.

        Called from the main thread, right before the kernel is verified and benchmarked. By default, this
        compiles the kernel, for backends that do not implement build().
        """
        return self.compile(kernel_instance)

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

    @abstractmethod
    def refresh_memory(self, device_memory, host_arguments, should_sync):
        """This method must implement refreshing the device memory with a clean copy."""
        pass


class GPUBackend(Backend):
    """Base class for GPU backends."""

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

    def refresh_memory(self, gpu_memory, host_arguments, should_sync):
        """Refresh the GPU memory with the untouched host arguments."""
        for i, arg in enumerate(host_arguments):
            if should_sync[i]:
                self.memcpy_htod(gpu_memory[i], arg)


class CompilerBackend(Backend):
    """Base class for compiler backends."""

    @abstractmethod
    def __init__(self, iterations, compiler_options, compiler):
        pass

    @abstractmethod
    def cleanup_lib(self):
        """Unload the previously loaded shared library"""
        pass
