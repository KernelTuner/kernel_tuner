from __future__ import print_function

try:
    from mock import patch
except ImportError:
    from unittest.mock import patch

import numpy
from kernel_tuner import core

from .test_interface import mock_config


@patch('kernel_tuner.core.CudaFunctions')
def test_check_kernel_output(dev_func_interface):
    dev_func_interface.configure_mock(**mock_config)

    dev = core.DeviceInterface(0, 0, "", lang="CUDA")
    dfi = dev.dev

    answer = [numpy.zeros(4).astype(numpy.float32)]
    instance = core.KernelInstance("name", "kernel_string", "temp_files", (256,1,1), (1,1,1), {}, answer)
    wrong = [numpy.array([1,2,3,4]).astype(numpy.float32)]
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

    answer = [numpy.zeros(4).astype(numpy.float32), None, numpy.ones(5).astype(numpy.int32)]

    answer_type_error1 = [numpy.zeros(4).astype(numpy.float32)]
    answer_type_error2 = [numpy.zeros(4).astype(numpy.float32), None, numpy.int32(1)]
    answer_type_error3 = [numpy.zeros(4).astype(numpy.float32), None, numpy.ones(4).astype(numpy.int32)]

    result_host = [numpy.zeros(4).astype(numpy.float32), None, numpy.ones(5).astype(numpy.int32)]
    result_host2 = [numpy.array([0,0,0,0]).reshape((2,2)).astype(numpy.float32), None, numpy.ones(5).astype(numpy.int32)]

    instance = core.KernelInstance("name", "kernel_string", [], (256,1,1), (1,1,1), {}, answer)

    for ans in [answer_type_error1, answer_type_error2, answer_type_error3]:
        try:
            core._default_verify_function(instance, ans, result_host, 0)
            print("check_kernel_output failed to throw an exception")
            assert False
        except TypeError:
            assert True

    for result_host in [result_host, result_host2]:
        assert core._default_verify_function(instance, answer, result_host, 0.1)


def test_default_verify_function_scalar():

    answer = [numpy.zeros(4).astype(numpy.float32), None, numpy.int64(42)]

    instance = core.KernelInstance("name", "kernel_string", [], (256,1,1), (1,1,1), {}, answer)

    answer_type_error1 = [numpy.zeros(4).astype(numpy.float32), None, numpy.float64(42)]
    answer_type_error2 = [numpy.zeros(4).astype(numpy.float32), None, numpy.float32(42)]

    result_host = [numpy.array([0,0,0,0]).astype(numpy.float32), None, numpy.int64(42)]

    for ans in [answer_type_error1, answer_type_error2]:
        try:
            core._default_verify_function(instance, ans, result_host, 0)
            print("check_kernel_output failed to throw an exception")
            assert False
        except TypeError:
            assert True

    assert core._default_verify_function(instance, answer, result_host, 0.1)

