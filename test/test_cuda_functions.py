import numpy as np
import pytest

from kernel_tuner import tune_kernel
from kernel_tuner.backends import nvcuda
from kernel_tuner.core import KernelInstance, KernelSource

from kernel_tuner.utils.nvcuda import cuda_error_check

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
def test_set_sm_percentage():

    dev = nvcuda.CudaFunctions(0)
    default_stream = dev.stream

    test_value = 50
    dev.set_sm_percentage(test_value)

    assert dev.current_sm_percentage == test_value
    assert test_value in dev.green_ctx_cache
    assert dev.green_ctx is not None
    assert not dev.stream == default_stream
    assert dev.assigned_sm_count


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
    err = nvcuda.runtime.cudaGetLastError()
    cuda_error_check(err)

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
    err = nvcuda.runtime.cudaGetLastError()
    cuda_error_check(err)

@skip_if_no_cuda
def test_tune_kernel(env):
    result, _ = tune_kernel(*env, lang="nvcuda", verbose=True)
    assert len(result) > 0

@skip_if_no_cuda
def test_copy_constant_memory_args():
    kernel_string = """
    __constant__ float my_constant_data[100];
    __global__ void copy_data_kernel(float* output) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < 100) {
            output[idx] = my_constant_data[idx];
        }
    }
    """

    kernel_name = "copy_data_kernel"
    kernel_sources = KernelSource(kernel_name, kernel_string, "NVCUDA")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])
    dev = nvcuda.CudaFunctions(0)
    kernel = dev.compile(kernel_instance)

    my_constant_data = np.full(100, 23).astype(np.float32)
    cmem_args = {'my_constant_data': my_constant_data}
    dev.copy_constant_memory_args(cmem_args)

    output = np.full(100, 0).astype(np.float32)
    gpu_args = dev.ready_argument_list([output])

    threads = (100, 1, 1)
    grid = (1, 1, 1)
    dev.run_kernel(kernel, gpu_args, threads, grid)

    dev.memcpy_dtoh(output, gpu_args[0])

    assert (my_constant_data == output).all()
