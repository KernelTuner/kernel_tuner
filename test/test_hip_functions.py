import numpy as np
import ctypes
from .context import skip_if_no_pyhip

import pytest
import kernel_tuner
from kernel_tuner.backends import hip as kt_hip
from kernel_tuner.core import KernelSource, KernelInstance

try: 
    from pyhip import hip, hiprtc
    hip_present = True
except ImportError:
    pass

@skip_if_no_pyhip
def test_ready_argument_list():

    size = 1000
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    c = np.bool_(True)
    d = np.zeros_like(b)

    arguments = [d, a, b, c]

    class ArgListStructure(ctypes.Structure):
        _fields_ = [("field0", ctypes.POINTER(ctypes.c_float)),
                    ("field1", ctypes.c_int),
                    ("field2", ctypes.POINTER(ctypes.c_float)),
                    ("field3", ctypes.c_bool)]
        def __getitem__(self, key):
                return self._fields_[key]

    dev = kt_hip.HipFunctions(0)
    gpu_args = dev.ready_argument_list(arguments)

    argListStructure = ArgListStructure(d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        ctypes.c_int(a),
                                        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        ctypes.c_bool(c))
    
    assert(gpu_args[1] == argListStructure[1])
    assert(gpu_args[3] == argListStructure[3])

@skip_if_no_pyhip
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


@skip_if_no_pyhip
def test_memset_and_memcpy_dtoh():
    a = [1, 2, 3, 4]
    x = np.array(a).astype(np.int8)
    x_d = hip.hipMalloc(x.nbytes)

    dev = kt_hip.HipFunctions()
    dev.memset(x_d, 4, x.nbytes)

    output = np.empty(4, dtype=np.int8)
    dev.memcpy_dtoh(output, x_d)

    assert all(output == np.full(4, 4))

@skip_if_no_pyhip
def test_memcpy_htod():
    a = [1, 2, 3, 4]
    x = np.array(a).astype(np.float32)
    x_d = hip.hipMalloc(x.nbytes)
    output = np.empty(4, dtype=np.float32)

    dev = kt_hip.HipFunctions()
    dev.memcpy_htod(x_d, x)
    dev.memcpy_dtoh(output, x_d)

    assert all(output == x)

@skip_if_no_pyhip
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

    dev.memcpy_dtoh(output, gpu_args.field0)

    assert(my_constant_data == output).all()

def dummy_func(a, b, block=0, grid=0, stream=None, shared=0, texrefs=None):
    pass

