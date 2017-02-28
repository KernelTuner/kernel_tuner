import numpy
from nose.tools import nottest
from .context import skip_if_no_cuda_device

from kernel_tuner import cuda

try:
    import pycuda.driver
except Exception:
    pass


def test_ready_argument_list():

    skip_if_no_cuda_device()

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

    gpu_args[0].free()
    gpu_args[2].free()


def test_compile():

    skip_if_no_cuda_device()

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

    kernel_string = original_kernel.replace("shared_size", str(100*1024*1024))

    dev = cuda.CudaFunctions(0)
    try:
        func = dev.compile("vector_add", kernel_string)
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
    try:
        func = dev.compile("vector_add", kernel_string)
        assert True
    except Exception as e:
        print("Did not expect any exception:")
        print(str(e))
        assert False

@nottest
def test_func(a, b, block=0, grid=0):
    pass

def test_benchmark():
    skip_if_no_cuda_device()
    dev = cuda.CudaFunctions(0)
    args = [1, 2]
    time = dev.benchmark(test_func, args, (1,2), (1,2))
    assert time > 0
