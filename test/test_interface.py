try:
    from mock import patch, call
except ImportError:
    from unittest.mock import patch

import numpy

from .context import *

mock_config = { "return_value.compile.return_value": "compile",
                "return_value.ready_argument_list.return_value": "ready_argument_list",
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

    expected = "#define block_size_x 128\n__global__ void fake_kernel_128()"
    dev.compile.assert_called_once_with("fake_kernel_128", expected)
    dev.benchmark.assert_called_once_with('compile', 'ready_argument_list', (128, 1, 1), (10, 1))

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface_handles_max_threads(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [256, 512] }
    dev.max_threads = 256

    kernel_tuner.tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, lang="CUDA")

    dev.compile.assert_called_once_with("fake_kernel_256", "#define block_size_x 256\nfake_kernel_256")

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface_handles_compile_error(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [256] }
    dev.compile.side_effect = Exception("uses too much shared data")

    kernel_tuner.tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, lang="CUDA")

    dev.compile.assert_called_once_with("fake_kernel_256", "#define block_size_x 256\nfake_kernel_256")
    assert dev.benchmark.called == False

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface_handles_restriction(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [128, 256, 512] }
    restrict = ["block_size_x > 128", "block_size_x < 512"]

    kernel_tuner.tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, restrictions=restrict, lang="CUDA", verbose=True)

    dev.compile.assert_called_once_with("fake_kernel_256", "#define block_size_x 256\nfake_kernel_256")
    dev.benchmark.assert_called_once_with('compile', 'ready_argument_list', (256, 1, 1), (1, 1))

@patch('kernel_tuner.interface.CudaFunctions')
def test_interface_handles_runtime_error(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [256] }
    dev.benchmark.side_effect = Exception("too many resources requested for launch")

    results = kernel_tuner.tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, lang="CUDA")

    dev.compile.assert_called_once_with("fake_kernel_256", "#define block_size_x 256\nfake_kernel_256")
    dev.benchmark.assert_called_once_with('compile', 'ready_argument_list', (256, 1, 1), (1, 1))
    assert len(results) == 0

@patch('kernel_tuner.interface.CudaFunctions')
def test_run_kernel(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    kernel_string = "__global__ void fake_kernel()"
    size = 1280
    problem_size = (size, 1)
    n = numpy.int32(size)
    args = [n]
    tune_params = dict()
    tune_params["block_size_x"] = 128
    kernel_tuner.run_kernel("fake_kernel", kernel_string, problem_size, args, tune_params)

    dev.compile.assert_called_once_with("fake_kernel", "#define block_size_x 128\n__global__ void fake_kernel()")
    dev.run_kernel.assert_called_once_with('compile', 'ready_argument_list', (128, 1, 1), (10, 1))
    dev.memcpy_dtoh.assert_called_once_with(numpy.zeros(1), 'r')

@patch('kernel_tuner.interface.CudaFunctions')
def test_check_kernel_correctness(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    answer = [numpy.zeros(8).astype(numpy.float32)]
    test = kernel_tuner._check_kernel_correctness(dev, 'func', ['gpu_args'], 'threads', 'grid', answer, 'instance_string')

    dev.memset.assert_called_once_with('gpu_args', 0, 8)
    dev.run_kernel.assert_called_once_with('func', ['gpu_args'], 'threads', 'grid')

    for name, args, _ in dev.mock_calls:
        if name == 'memcpy_dtoh':
            assert all(args[0] == answer[0])
            assert args[1] == 'gpu_args'
    assert dev.memcpy_dtoh.called == 1
    assert test
