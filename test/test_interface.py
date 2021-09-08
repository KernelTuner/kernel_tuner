from __future__ import print_function

try:
    from mock import patch
except ImportError:
    from unittest.mock import patch

import numpy as np

from kernel_tuner.interface import tune_kernel, run_kernel

mock_config = {"return_value.compile.return_value": "compile",
               "return_value.ready_argument_list.return_value": "ready_argument_list",
               "return_value.max_threads": 1024}


def get_fake_kernel():
    kernel_string = "__global__ void fake_kernel(int number)"
    size = 1280
    n = np.int32(size)
    args = [n]
    tune_params = {"block_size_x": [128]}
    return "fake_kernel", kernel_string, size, args, tune_params


@patch('kernel_tuner.core.CudaFunctions')
def test_interface_calls_functions(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    kernel_name, kernel_string, size, args, tune_params = get_fake_kernel()
    tune_kernel(kernel_name, kernel_string, size, args, tune_params, verbose=True)

    dev.compile.assert_called()
    dev.benchmark.assert_called_with('compile', 'ready_argument_list', (128, 1, 1), (10, 1, 1))


@patch('kernel_tuner.core.CudaFunctions')
def test_interface_handles_max_threads(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = {"block_size_x": [256, 512]}
    dev.max_threads = 256

    kernel_string = "__global__ void fake_kernel(int number)"
    tune_kernel("fake_kernel", kernel_string, (1, 1), [np.int32(0)], tune_params, lang="CUDA")

    # verify that only a single instance of the kernel is compiled
    dev.compile.assert_called()


@patch('kernel_tuner.core.CudaFunctions')
def test_interface_handles_compile_error(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = {"block_size_x": [256]}
    dev.compile.side_effect = Exception("uses too much shared data")

    kernel_string = "__global__ void fake_kernel(int number)"
    tune_kernel("fake_kernel", kernel_string, (1, 1), [np.int32(0)], tune_params, lang="CUDA")

    assert dev.compile.call_count == 2
    assert not dev.benchmark.called


@patch('kernel_tuner.core.CudaFunctions')
def test_interface_handles_restriction(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = {"block_size_x": [128, 256, 512]}
    restrict = ["block_size_x > 128", "block_size_x < 512"]

    kernel_string = "__global__ void fake_kernel(int number)"
    tune_kernel("fake_kernel", kernel_string, (1, 1), [np.int32(0)], tune_params, restrictions=restrict,
                lang="CUDA", verbose=True)

    assert dev.compile.call_count == 2
    dev.benchmark.assert_called_with('compile', 'ready_argument_list', (256, 1, 1), (1, 1, 1))


@patch('kernel_tuner.core.CudaFunctions')
def test_interface_handles_runtime_error(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = {"block_size_x": [256]}
    dev.benchmark.side_effect = Exception("too many resources requested for launch")

    kernel_string = "__global__ void fake_kernel(int number)"
    results, _ = tune_kernel("fake_kernel", kernel_string, (1, 1), [np.int32(0)], tune_params, lang="CUDA")

    assert dev.compile.call_count == 2
    dev.benchmark.assert_called_with('compile', 'ready_argument_list', (256, 1, 1), (1, 1, 1))
    assert len(results) == 0


@patch('kernel_tuner.core.CudaFunctions')
def test_run_kernel(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    kernel_name, kernel_string, size, args, _ = get_fake_kernel()
    answer = run_kernel(kernel_name, kernel_string, size, args, {"block_size_x": 128})

    assert dev.compile.call_count == 1
    dev.run_kernel.assert_called_with('compile', 'ready_argument_list', (128, 1, 1), (10, 1, 1))
    assert answer[0] == size
