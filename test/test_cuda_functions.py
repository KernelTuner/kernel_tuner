import numpy as np
import pytest

from kernel_tuner import tune_kernel
from kernel_tuner.backends import nvcuda
from kernel_tuner.core import KernelInstance, KernelSource

from .context import skip_if_no_cuda
from .test_runners import env  # noqa: F401

try:
    from cuda.bindings import driver
except Exception:
    pass


@skip_if_no_cuda
def test_ready_argument_list():

    size = 1000
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)

    arguments = [c, a, b]

    dev = nvcuda.CudaFunctions(0)
    gpu_args = dev.ready_argument_list(arguments)

    assert isinstance(gpu_args[0], driver.CUdeviceptr)
    assert isinstance(gpu_args[1], np.int32)
    assert isinstance(gpu_args[2], driver.CUdeviceptr)


def create_kernel_instance(kernel_name, kernel_string):
    kernel_sources = KernelSource(kernel_name, kernel_string, "cuda")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])
    return kernel_instance


@skip_if_no_cuda
def test_compile():

    kernel_string = """
    extern "C" __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    kernel_instance = create_kernel_instance("vector_add", kernel_string)
    dev = nvcuda.CudaFunctions(0)
    dev.compile(kernel_instance)

@skip_if_no_cuda
def test_compile_template():

    kernel_string = """
    namespace nested::namespaces {
    template <typename T, int N>
    __global__ void vector_add(T *c, T *a, T *b) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<N) {
            c[i] = a[i] + b[i];
        }
    }
    }
    """

    kernel_name = "nested::namespaces::vector_add<float,10>"
    kernel_instance = create_kernel_instance(kernel_name, kernel_string)
    dev = nvcuda.CudaFunctions(0, compiler_options=["-std=c++17"])
    dev.compile(kernel_instance)

@skip_if_no_cuda
def test_compile_include():

    kernel_string = """
    #include <cuda_fp16.h>

    __global__ void vector_add(__nv_half *c, __nv_half *a, __nv_half *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = __hadd(a[i], b[i]);
        }
    }
    """

    kernel_instance = create_kernel_instance("vector_add", kernel_string)
    dev = nvcuda.CudaFunctions(0, compiler_options=["-std=c++17"])
    dev.compile(kernel_instance)

@skip_if_no_cuda
def test_tune_kernel(env):
    result, _ = tune_kernel(*env, lang="nvcuda", verbose=True)
    assert len(result) > 0
