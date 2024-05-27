from .context import (
    skip_if_no_gcc,
    skip_if_no_cupy,
    skip_if_no_cuda,
    skip_if_no_opencl,
    skip_if_no_pycuda,
)
from kernel_tuner.backends import backend, compiler, cupy, nvcuda, opencl, pycuda


class WrongBackend(backend.Backend):
    """This is a not compliant backend"""

    pass


def test_wrong_backend():
    try:
        dev = WrongBackend()
        assert False
    except TypeError:
        assert True


@skip_if_no_gcc
def test_c_backend():
    dev = compiler.CompilerFunctions()


@skip_if_no_cupy
def test_cupy_backend():
    dev = cupy.CupyFunctions()


@skip_if_no_cuda
def test_cuda_backend():
    dev = nvcuda.CudaFunctions()


@skip_if_no_opencl
def test_opencl_backend():
    dev = opencl.OpenCLFunctions()


@skip_if_no_pycuda
def test_pycuda_backend():
    dev = pycuda.PyCudaFunctions()
