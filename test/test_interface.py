from mock import patch
import numpy

from .context import *

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface1(dev_interface):
    dev = dev_interface.return_value

    dev.compile.return_value = "compile"
    dev.create_gpu_args.return_value = "create_gpu_args"
    dev.max_threads = 1024
    kernel_string = "__global__ void fake_kernel()"

    size = 1280
    problem_size = (size, 1)
    n = numpy.int32(size)
    args = [n]
    tune_params = dict()
    tune_params["block_size_x"] = [128]
    kernel_tuner.tune_kernel("fake_kernel", kernel_string, problem_size, args, tune_params, verbose=True)

    print dev_interface.mock_calls

    dev.compile.assert_called_once_with("fake_kernel_128", kernel_string.replace('fake_kernel', 'fake_kernel_128'))
    dev.benchmark.assert_called_once_with('compile', 'create_gpu_args', (128, 1, 1), (10, 1))



