"""This module contains the interface of all kernel_tuner backends"""
from __future__ import print_function

from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def __init__(self, device, iterations, compiler_options, observers):
        pass

    @abstractmethod
    def __del__(self):
        pass

    @abstractmethod
    def ready_argument_list(self, arguments):
        pass

    @abstractmethod
    def compile(self, kernel_instance):
        pass

    @abstractmethod
    def start_event(self):
        pass

    @abstractmethod
    def stop_event(self):
        pass

    @abstractmethod
    def kernel_finished(self):
        pass

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def copy_constant_memory_args(self, cmem_args):
        pass

    @abstractmethod
    def copy_shared_memory_args(self, smem_args):
        pass

    @abstractmethod
    def copy_texture_memory_args(self, texmem_args):
        pass

    @abstractmethod
    def run_kernel(self, func, gpu_args, threads, grid, stream):
        pass

    @abstractmethod
    def memset(self, allocation, value, size):
        pass

    @abstractmethod
    def memcpy_dtoh(self, dest, src):
        pass

    @abstractmethod
    def memcpy_htod(self, dest, src):
        pass
