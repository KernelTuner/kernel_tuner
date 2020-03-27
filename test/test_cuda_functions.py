import numpy
from .context import skip_if_no_cuda

from kernel_tuner import cuda
from kernel_tuner.core import KernelSource, KernelInstance

try:
    import pycuda.driver
except Exception:
    pass

@skip_if_no_cuda
def test_ready_argument_list():

    size = 1000
    a = numpy.int32(75)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)

    arguments = [c, a, b]

    dev = cuda.CudaFunctions(0)
    gpu_args = dev.ready_argument_list(arguments)

    assert isinstance(gpu_args[0], pycuda.driver.DeviceAllocation)
    assert isinstance(gpu_args[1], numpy.int32)
    assert isinstance(gpu_args[2], pycuda.driver.DeviceAllocation)

@skip_if_no_cuda
def test_compile():

    original_kernel = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        __shared__ float test[shared_size];
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            test[0] = a[i];
            c[i] = test[0] + b[i];
        }
    }
    """

    kernel_sources = KernelSource(original_kernel, "cuda")


    kernel_string = original_kernel.replace("shared_size", str(100*1024*1024))
    kernel_instance = KernelInstance("vector_add", kernel_sources, kernel_string, [], None, None, dict(), [])

    dev = cuda.CudaFunctions(0)
    try:
        func = dev.compile(kernel_instance)
        assert isinstance(func, pycuda.driver.Function)
        print("Expected an exception because too much shared memory is requested")
        assert False
    except Exception as e:
        if "uses too much shared data" in str(e):
            assert True
        else:
            print("Expected a different exception:" + str(e))
            assert False

    kernel_string = original_kernel.replace("shared_size", str(100))
    kernel_instance = KernelInstance("vector_add", kernel_sources, kernel_string, [], None, None, dict(), [])
    try:
        func = dev.compile( kernel_instance)
        assert True
    except Exception as e:
        print("Did not expect any exception:")
        print(str(e))
        assert False

def dummy_func(a, b, block=0, grid=0, stream=None, texrefs=None):
    pass

@skip_if_no_cuda
def test_benchmark():
    dev = cuda.CudaFunctions(0)
    args = [1, 2]
    res = dev.benchmark(dummy_func, args, (1,2), (1,2))
    assert res["time"] > 0
    assert len(res["times"]) == dev.iterations
