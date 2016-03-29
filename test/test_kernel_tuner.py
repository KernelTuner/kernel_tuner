import numpy
import pycuda.driver

from .context import kernel_tuner

def test_create_gpu_args():

    size = 1000
    a = 0.75
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)

    arguments = [c, a, b]

    gpu_args = kernel_tuner._create_gpu_args(arguments)

    assert type(gpu_args[0]) is pycuda.driver.DeviceAllocation
    assert type(gpu_args[1]) is float
    assert type(gpu_args[2]) is pycuda.driver.DeviceAllocation

    gpu_args[0].free()
    gpu_args[2].free()

