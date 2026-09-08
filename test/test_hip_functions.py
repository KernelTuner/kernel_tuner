import ctypes
import numpy as np
import pytest

from kernel_tuner import tune_kernel
from kernel_tuner.backends import hip as kt_hip
from kernel_tuner.core import KernelInstance, KernelSource

from .context import skip_if_no_hip

try:
    from hip import hip, hiprtc
    hip_present = True
except ImportError:
    pass

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif isinstance(err, hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        raise RuntimeError(str(err))
    return result

@pytest.fixture
def env():
    kernel_string = """
    extern "C" __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 100
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    tune_params = dict()
    tune_params["block_size_x"] = [128 + 64 * i for i in range(15)]

    return ["vector_add", kernel_string, size, args, tune_params]

@skip_if_no_hip
def test_ready_argument_list():
    size = 1000
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    c = np.bool_(True)
    d = np.zeros_like(b)

    arguments = [d, a, b, c]

    dev = kt_hip.HipFunctions(0)
    gpu_args = dev.ready_argument_list(arguments)

    # ctypes have no equality defined, so indirect comparison for type and value
    assert isinstance(gpu_args[1], ctypes.c_int)
    assert isinstance(gpu_args[3], ctypes.c_bool)
    assert gpu_args[1].value == a
    assert gpu_args[3].value == c

@skip_if_no_hip
def test_compile():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    kernel_name = "vector_add"
    kernel_sources = KernelSource(kernel_name, kernel_string, "HIP")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])
    dev = kt_hip.HipFunctions(0)
    try:
        dev.compile(kernel_instance)
    except Exception as e:
        pytest.fail("Did not expect any exception:" + str(e))

@skip_if_no_hip
def test_memset_and_memcpy_dtoh():
    a = [1, 2, 3, 4]
    x = np.array(a).astype(np.int8)
    x_d = hip_check(hip.hipMalloc(x.nbytes))

    dev = kt_hip.HipFunctions()
    dev.memset(x_d, 4, x.nbytes)

    output = np.empty(4, dtype=np.int8)
    dev.memcpy_dtoh(output, x_d)

    assert all(output == np.full(4, 4))

@skip_if_no_hip
def test_memcpy_htod():
    a = [1, 2, 3, 4]
    x = np.array(a).astype(np.float32)
    x_d = hip_check(hip.hipMalloc(x.nbytes))
    output = np.empty(4, dtype=np.float32)

    dev = kt_hip.HipFunctions()
    dev.memcpy_htod(x_d, x)
    dev.memcpy_dtoh(output, x_d)

    assert all(output == x)

@skip_if_no_hip
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
    kernel_sources = KernelSource(kernel_name, kernel_string, "HIP")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])
    dev = kt_hip.HipFunctions(0)
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

@skip_if_no_hip
def test_smem_args(env):
    result, _ = tune_kernel(*env,
                          smem_args=dict(size="block_size_x*4"),
                          verbose=True, lang="HIP")
    tune_params = env[-1]
    assert len(result) == len(tune_params["block_size_x"])
    result, _ = tune_kernel(
        *env,
        smem_args=dict(size=lambda p: p['block_size_x'] * 4),
        verbose=True, lang="HIP")
    tune_params = env[-1]
    assert len(result) == len(tune_params["block_size_x"])