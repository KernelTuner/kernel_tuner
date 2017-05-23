from __future__ import print_function

try:
    from mock import patch
except ImportError:
    from unittest.mock import patch

import numpy

from kernel_tuner.interface import tune_kernel, run_kernel
from kernel_tuner import core

mock_config = { "return_value.compile.return_value": "compile",
                "return_value.ready_argument_list.return_value": "ready_argument_list",
                "return_value.max_threads": 1024 }

@patch('kernel_tuner.core.CudaFunctions')
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
    tune_kernel("fake_kernel", kernel_string, problem_size, args, tune_params, verbose=True)

    expected = "#define block_size_x 128\n#define grid_size_z 1\n#define grid_size_y 1\n#define grid_size_x 10\n__global__ void fake_kernel_128()"
    dev.compile.assert_called_once_with("fake_kernel_128", expected)
    dev.benchmark.assert_called_once_with('compile', 'ready_argument_list', (128, 1, 1), (10, 1, 1))

@patch('kernel_tuner.core.CudaFunctions')
def test_interface_handles_max_threads(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [256, 512] }
    dev.max_threads = 256

    tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, lang="CUDA")

    dev.compile.assert_called_once_with("fake_kernel_256", "#define block_size_x 256\n#define grid_size_z 1\n#define grid_size_y 1\n#define grid_size_x 1\nfake_kernel_256")

@patch('kernel_tuner.core.CudaFunctions')
def test_interface_handles_compile_error(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [256] }
    dev.compile.side_effect = Exception("uses too much shared data")

    tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, lang="CUDA")

    assert dev.compile.call_count == 1
    assert dev.benchmark.called == False

@patch('kernel_tuner.core.CudaFunctions')
def test_interface_handles_restriction(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [128, 256, 512] }
    restrict = ["block_size_x > 128", "block_size_x < 512"]

    tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, restrictions=restrict, lang="CUDA", verbose=True)

    assert dev.compile.call_count == 1
    dev.benchmark.assert_called_once_with('compile', 'ready_argument_list', (256, 1, 1), (1, 1, 1))

@patch('kernel_tuner.core.CudaFunctions')
def test_interface_handles_runtime_error(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    tune_params = { "block_size_x": [256] }
    dev.benchmark.side_effect = Exception("too many resources requested for launch")

    results, _ = tune_kernel("fake_kernel", "fake_kernel", (1,1), [numpy.int32(0)], tune_params, lang="CUDA")

    assert dev.compile.call_count == 1
    dev.benchmark.assert_called_once_with('compile', 'ready_argument_list', (256, 1, 1), (1, 1, 1))
    assert len(results) == 0

@patch('kernel_tuner.core.CudaFunctions')
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
    answer = run_kernel("fake_kernel", kernel_string, problem_size, args, tune_params)

    assert dev.compile.call_count == 1
    dev.run_kernel.assert_called_once_with('compile', 'ready_argument_list', (128, 1, 1), (10, 1, 1))
    assert answer[0] == n

@patch('kernel_tuner.core.CudaFunctions')
def test_check_kernel_correctness(dev_interface):
    dev = dev_interface.return_value
    dev_interface.configure_mock(**mock_config)

    answer = [numpy.zeros(4).astype(numpy.float32)]
    wrong = [numpy.array([1,2,3,4]).astype(numpy.float32)]

    instance = {'threads': 'threads', 'grid': 'grid', 'params': {}}
    test = core.check_kernel_correctness(dev, 'func', answer, instance, answer, True)

    dev.memset.assert_called_once_with(answer[0], 0, answer[0].nbytes)
    dev.run_kernel.assert_called_once_with('func', answer, 'threads', 'grid')

    print(dev.mock_calls)

    assert dev.memcpy_dtoh.called == 1

    for name, args, _ in dev.mock_calls:
        if name == 'memcpy_dtoh':
            assert all(args[0] == answer[0])
            assert all(args[1] == answer[0])
    assert test

    #the following call to check_kernel_correctness is expected to fail because
    #the answer is non-zero, while the memcpy_dtoh function on the Mocked object
    #obviously does not result in the result_host array containing anything
    #non-zero
    try:
        core.check_kernel_correctness(dev, 'func', wrong, instance, wrong, True)
        print("check_kernel_correctness failed to throw an exception")
        assert False
    except Exception as e:
        assert True

