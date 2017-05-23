from __future__ import print_function

import numpy
import ctypes as C
from nose.tools import raises

try:
    from mock import patch, Mock
except ImportError:
    from unittest.mock import patch, Mock

from kernel_tuner.c import CFunctions


def test_ready_argument_list1():
    arg1 = numpy.array([1, 2, 3]).astype(numpy.float32)
    arg2 = numpy.array([4, 5, 6]).astype(numpy.float64)
    arg3 = numpy.array([7, 8, 9]).astype(numpy.int32)
    arguments = [arg1, arg2, arg3]

    cfunc = CFunctions();

    output = cfunc.ready_argument_list(arguments)
    print(output)

    output_arg1 = numpy.ctypeslib.as_array(output[0], shape=arg1.shape)
    output_arg2 = numpy.ctypeslib.as_array(output[1], shape=arg2.shape)
    output_arg3 = numpy.ctypeslib.as_array(output[2], shape=arg3.shape)

    assert output_arg1.dtype == 'float32'
    assert output_arg2.dtype == 'float64'
    assert output_arg3.dtype == 'int32'

    assert all(output_arg1 == arg1)
    assert all(output_arg2 == arg2)
    assert all(output_arg3 == arg3)

def test_ready_argument_list2():
    arg1 = numpy.array([1, 2, 3]).astype(numpy.float32)
    arg2 = numpy.int32(7)
    arg3 = numpy.float32(6.0)
    arguments = [arg1, arg2, arg3]

    cfunc = CFunctions()
    output = cfunc.ready_argument_list(arguments)
    print(output)

    output_arg1 = numpy.ctypeslib.as_array(output[0], shape=arg1.shape)

    assert output_arg1.dtype == 'float32'
    assert isinstance(output[1], C.c_int32)
    assert isinstance(output[2], C.c_float)

    assert all(output_arg1 == arg1)
    assert output[1].value == arg2
    assert output[2].value == arg3

def test_ready_argument_list3():
    arg1 = Mock()
    arguments = [arg1]
    cfunc = CFunctions()
    try:
        cfunc.ready_argument_list(arguments)
        assert False
    except Exception:
        assert True

@raises(TypeError)
def test_ready_argument_list4():
    arg1 = int(9)
    cfunc = CFunctions()
    cfunc.ready_argument_list([arg1])


@patch('kernel_tuner.c.subprocess')
@patch('kernel_tuner.c.numpy.ctypeslib')
def test_compile(npct, subprocess):

    kernel_string = "this is a fake C program"
    kernel_name = "blabla"

    cfunc = CFunctions()
    f = cfunc.compile(kernel_name, kernel_string)

    print(subprocess.mock_calls)
    print(npct.mock_calls)
    print(f)

    assert len(subprocess.mock_calls) == 6
    assert npct.load_library.called == 1

    args, _ = npct.load_library.call_args_list[0]
    filename = args[0]
    print('filename=' + filename)

    #check if temporary files are cleaned up correctly
    import os.path
    assert not os.path.isfile(filename + ".cu")
    assert not os.path.isfile(filename + ".o")
    assert not os.path.isfile(filename + ".so")


def test_memset():
    a = [1, 2, 3, 4]
    x = numpy.array(a).astype(numpy.float32)
    x_c = x.ctypes.data_as(C.POINTER(C.c_float))

    cfunc = CFunctions()
    cfunc.memset(x_c, 0, x.nbytes)

    output = numpy.ctypeslib.as_array(x_c, shape=(4,))

    print(output)
    assert all(output == numpy.zeros(4))

def test_memcpy_dtoh():
    a = [1, 2, 3, 4]
    x = numpy.array(a).astype(numpy.float32)
    x_c = x.ctypes.data_as(C.POINTER(C.c_float))
    output = numpy.zeros_like(x)

    cfunc = CFunctions()
    cfunc.arg_mapping = { str(x_c) : (4,) }
    cfunc.memcpy_dtoh(output, x_c)

    print(a)
    print(output)

    assert all(output == a)
