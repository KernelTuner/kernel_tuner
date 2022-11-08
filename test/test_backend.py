import pytest

import kernel_tuner
from .context import skip_if_no_gcc, skip_if_no_cupy, skip_if_no_cuda, skip_if_no_opencl, skip_if_no_pycuda
from kernel_tuner.backends import backend, c, cupy, nvcuda, opencl, pycuda


class WrongBackend(backend.Backend):
    """This is a not compliant backend"""
    pass


def test_wrong_backend():
    try:
        dev = WrongBackend()
        assert False
    except TypeError:
        assert True


def test_c_backend():
    dev = c.CFunctions()

def test_cupy_backend():
    dev = cupy.CupyFunctions()

def test_cuda_backend():
    dev = nvcuda.CudaFunctions()

def test_opencl_backend():
    dev = opencl.OpenCLFunctions()

def test_pycuda_backend():
    dev = pycuda.PyCudaFunctions()