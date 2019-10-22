from __future__ import print_function

import numpy as np
import pytest

try:
    from mock import patch
except ImportError:
    from unittest.mock import patch

import kernel_tuner
from kernel_tuner import core, util
from kernel_tuner.interface import Options

from .context import skip_if_no_cuda
from .test_interface import mock_config

def get_vector_add_args():
    size = int(1e6)
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b).astype(np.float32)
    n = np.int32(size)
    return c, a, b, n

@pytest.fixture
def env():
    kernel_string = """__global__ void vector_add(float *c, float *a, float *b, int n) {
            int i = blockIdx.x * block_size_x + threadIdx.x;
            if (i<n) {
                c[i] = a[i] + b[i];
            }
        } """
    args = get_vector_add_args()
    params = {"block_size_x": 128}

    lang="CUDA"
    kernel_source = core.KernelSource(kernel_string, lang)
    verbose = True
    kernel_options = Options(kernel_name="vector_add", kernel_string=kernel_string, problem_size=args[-1], arguments=args, lang=lang,
                          grid_div_x=None, grid_div_y=None, grid_div_z=None, cmem_args=None, texmem_args=None, block_size_names=None)
    device_options = Options(device=0, platform=0, lang=lang, quiet=False, compiler=None, compiler_options=None)
    dev = core.DeviceInterface(kernel_source, iterations=7, **device_options)
    instance = dev.create_kernel_instance(kernel_source, kernel_options, params, verbose)

    return dev, instance


@skip_if_no_cuda
def test_default_verify_function(env):

    #gpu_args = dev.ready_argument_list(args)
    #func = dev.compile_kernel(instance, verbose)

    dev, instance = env
    args = instance.arguments
    verbose = True

    #1st case, correct answer but not enough items in the list
    answer = [args[1] + args[2]]
    try:
        core._default_verify_function(instance, answer, args, 1e-6, verbose)
        #dev.check_kernel_output(func, gpu_args, instance, answer, 1e-6, None, verbose)
        print("Expected a TypeError to be raised")
        assert False
    except TypeError as expected_error:
        print(str(expected_error))
        assert "The length of argument list and provided results do not match." == str(expected_error)
    except Exception:
        print("Expected a TypeError to be raised")
        assert False

    #2nd case, answer is of wrong type
    answer = [np.ubyte([12]), None, None, None]
    try:
        core._default_verify_function(instance, answer, args, 1e-6, verbose)
        #dev.check_kernel_output(func, gpu_args, instance, answer, 1e-6, None, verbose)
        print("Expected a TypeError to be raised")
        assert False
    except TypeError as expected_error:
        print(str(expected_error))
        assert "Element 0" in str(expected_error)
    except Exception:
        print("Expected a TypeError to be raised")
        assert False

    instance.delete_temp_files()
    assert True


@patch('kernel_tuner.core.CudaFunctions')
def test_check_kernel_output(dev_func_interface):
    dev_func_interface.configure_mock(**mock_config)

    dev = core.DeviceInterface(core.KernelSource("", lang="CUDA"))
    dfi = dev.dev

    answer = [np.zeros(4).astype(np.float32)]
    instance = core.KernelInstance("name", None, "kernel_string", "temp_files", (256,1,1), (1,1,1), {}, answer)
    wrong = [np.array([1,2,3,4]).astype(np.float32)]
    atol = 1e-6

    test = dev.check_kernel_output('func', answer, instance, answer, atol, None, True)

    dfi.memcpy_htod.assert_called_once_with(answer[0], answer[0])
    dfi.run_kernel.assert_called_once_with('func', answer, (256,1,1), (1,1,1))

    print(dfi.mock_calls)

    assert dfi.memcpy_dtoh.called == 1

    for name, args, _ in dfi.mock_calls:
        if name == 'memcpy_dtoh':
            assert all(args[0] == answer[0])
            assert all(args[1] == answer[0])
    assert test

    #the following call to check_kernel_output is expected to fail because
    #the answer is non-zero, while the memcpy_dtoh function on the Mocked object
    #obviously does not result in the result_host array containing anything
    #non-zero
    try:
        dev.check_kernel_output('func', wrong, instance, wrong, atol, None, True)
        print("check_kernel_output failed to throw an exception")
        assert False
    except Exception:
        assert True



def test_default_verify_function_arrays():

    answer = [np.zeros(4).astype(np.float32), None, np.ones(5).astype(np.int32)]

    answer_type_error1 = [np.zeros(4).astype(np.float32)]
    answer_type_error2 = [np.zeros(4).astype(np.float32), None, np.int32(1)]
    answer_type_error3 = [np.zeros(4).astype(np.float32), None, np.ones(4).astype(np.int32)]

    result_host = [np.zeros(4).astype(np.float32), None, np.ones(5).astype(np.int32)]
    result_host2 = [np.array([0,0,0,0]).reshape((2,2)).astype(np.float32), None, np.ones(5).astype(np.int32)]

    instance = core.KernelInstance("name", None, "kernel_string", [], (256,1,1), (1,1,1), {}, answer)

    for ans in [answer_type_error1, answer_type_error2, answer_type_error3]:
        try:
            core._default_verify_function(instance, ans, result_host, 0, False)
            print("_default_verify_function failed to throw an exception")
            assert False
        except TypeError:
            assert True

    for result_host in [result_host, result_host2]:
        assert core._default_verify_function(instance, answer, result_host, 0.1, False)


def test_default_verify_function_scalar():

    answer = [np.zeros(4).astype(np.float32), None, np.int64(42)]

    instance = core.KernelInstance("name", None, "kernel_string", [], (256,1,1), (1,1,1), {}, answer)

    answer_type_error1 = [np.zeros(4).astype(np.float32), None, np.float64(42)]
    answer_type_error2 = [np.zeros(4).astype(np.float32), None, np.float32(42)]

    result_host = [np.array([0,0,0,0]).astype(np.float32), None, np.int64(42)]

    for ans in [answer_type_error1, answer_type_error2]:
        try:
            core._default_verify_function(instance, ans, result_host, 0, False)
            print("_default_verify_function failed to throw an exception")
            assert False
        except TypeError:
            assert True

    assert core._default_verify_function(instance, answer, result_host, 0.1, False)

