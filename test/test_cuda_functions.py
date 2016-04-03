import numpy
from nose import SkipTest
from nose.tools import nottest, raises
from .context import kernel_tuner
from .context import cuda

try:
    import pycuda.driver
except Exception:
    pass

@nottest
def skip_if_no_cuda_device():
    try:
        from pycuda.autoinit import context
    #except pycuda.driver.RuntimeError, e:
    except Exception, e:
        if "No module named pycuda.autoinit" in str(e):
            raise SkipTest("PyCuda not installed")
        elif "no CUDA-capable device is detected" in str(e):
            raise SkipTest("no CUDA-capable device is detected")
        else:
            raise e

def test_create_gpu_args():

    skip_if_no_cuda_device()

    size = 1000
    a = numpy.int32(75)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)

    arguments = [c, a, b]

    dev = cuda.CudaFunctions(0)
    gpu_args = dev.create_gpu_args(arguments)

    assert isinstance(gpu_args[0], pycuda.driver.DeviceAllocation)
    assert isinstance(gpu_args[1], numpy.int32)
    assert isinstance(gpu_args[2], pycuda.driver.DeviceAllocation)

    gpu_args[0].free()
    gpu_args[2].free()


