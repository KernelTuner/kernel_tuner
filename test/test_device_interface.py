from __future__ import print_function

try:
    from mock import patch
except ImportError:
    from unittest.mock import patch

import numpy
from kernel_tuner import core

from .test_interface import mock_config

@patch('kernel_tuner.core.CudaFunctions')
def test_check_kernel_correctness(dev_func_interface):
    dev_func_interface.configure_mock(**mock_config)

    dev = core.DeviceInterface(0, 0, "", lang="CUDA")
    dfi = dev.dev

    answer = [numpy.zeros(4).astype(numpy.float32)]
    wrong = [numpy.array([1,2,3,4]).astype(numpy.float32)]
    atol = 1e-6

    instance = core.KernelInstance("name", "kernel_string", "temp_files", (256,1,1), (1,1,1), {}, answer)
    test = dev.check_kernel_correctness('func', answer, instance, answer, atol, None, True)

    dfi.memset.assert_called_once_with(answer[0], 0, answer[0].nbytes)
    dfi.run_kernel.assert_called_once_with('func', answer, (256,1,1), (1,1,1))

    print(dfi.mock_calls)

    assert dfi.memcpy_dtoh.called == 1

    for name, args, _ in dfi.mock_calls:
        if name == 'memcpy_dtoh':
            assert all(args[0] == answer[0])
            assert all(args[1] == answer[0])
    assert test

    #the following call to check_kernel_correctness is expected to fail because
    #the answer is non-zero, while the memcpy_dtoh function on the Mocked object
    #obviously does not result in the result_host array containing anything
    #non-zero
    try:
        dev.check_kernel_correctness('func', wrong, instance, wrong, atol, None, True)
        print("check_kernel_correctness failed to throw an exception")
        assert False
    except Exception:
        assert True

