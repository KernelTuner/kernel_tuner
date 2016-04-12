from mock import patch, call
import numpy

from .context import *

mock_config = { "return_value.compile.return_value": "compile",
                "return_value.create_gpu_args.return_value": "create_gpu_args",
                "return_value.max_threads": 1024 }

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface_calls_functions(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    kernel_string = "__global__ void fake_kernel()"
    size = 1280
    problem_size = (size, 1)
    n = numpy.int32(size)
    args = [n]
    tune_params = dict()
    tune_params["block_size_x"] = [128]
    kernel_tuner.tune_kernel("fake_kernel", kernel_string, problem_size, args, tune_params, verbose=True)

    dev.compile.assert_called_once_with("fake_kernel_128", kernel_string.replace('fake_kernel', 'fake_kernel_128'))
    dev.benchmark.assert_called_once_with('compile', 'create_gpu_args', (128, 1, 1), (10, 1))

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface_handles_max_threads(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [128, 256, 512] }
    dev.max_threads = 256

    kernel_tuner.tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, lang="CUDA")

    calls = [call("fake_kernel_128", 'fake_kernel_128'), call("fake_kernel_256", 'fake_kernel_256')]
    dev.compile.assert_has_calls(calls)
    assert dev.compile.call_count == 2

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface_handles_compile_error(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [256] }
    dev.compile.side_effect = Exception("uses too much shared data")

    kernel_tuner.tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, lang="CUDA")

    dev.compile.assert_called_once_with("fake_kernel_256", 'fake_kernel_256')
    assert dev.benchmark.called == False

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface_handles_restriction(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [128, 256, 512] }
    restrict = ["block_size_x > 128", "block_size_x < 512"]

    kernel_tuner.tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, restrictions=restrict, lang="CUDA")

    dev.compile.assert_called_once_with("fake_kernel_256", 'fake_kernel_256')
    dev.benchmark.assert_called_once_with('compile', 'create_gpu_args', (256, 1, 1), (1, 1))

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface_handles_runtime_error(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [256] }
    dev.benchmark.side_effect = Exception("too many resources requested for launch")

    results = kernel_tuner.tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, lang="CUDA")

    dev.compile.assert_called_once_with("fake_kernel_256", 'fake_kernel_256')
    dev.benchmark.assert_called_once_with('compile', 'create_gpu_args', (256, 1, 1), (1, 1))
    assert len(results) == 0




