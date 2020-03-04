from __future__ import print_function

import numpy
import ctypes as C
from pytest import raises

try:
    from mock import patch, Mock
except ImportError:
    from unittest.mock import patch, Mock

from kernel_tuner.c import CFunctions, Argument
from kernel_tuner.core import KernelSource, KernelInstance


def test_ready_argument_list1():
    arg1 = numpy.array([1, 2, 3]).astype(numpy.float32)
    arg2 = numpy.array([4, 5, 6]).astype(numpy.float64)
    arg3 = numpy.array([7, 8, 9]).astype(numpy.int32)
    arguments = [arg1, arg2, arg3]

    cfunc = CFunctions()

    output = cfunc.ready_argument_list(arguments)
    print(output)

    output_arg1 = numpy.ctypeslib.as_array(output[0].ctypes, shape=arg1.shape)
    output_arg2 = numpy.ctypeslib.as_array(output[1].ctypes, shape=arg2.shape)
    output_arg3 = numpy.ctypeslib.as_array(output[2].ctypes, shape=arg3.shape)

    assert output_arg1.dtype == 'float32'
    assert output_arg2.dtype == 'float64'
    assert output_arg3.dtype == 'int32'

    assert all(output_arg1 == arg1)
    assert all(output_arg2 == arg2)
    assert all(output_arg3 == arg3)

    assert output[0].numpy.dtype == 'float32'
    assert output[1].numpy.dtype == 'float64'
    assert output[2].numpy.dtype == 'int32'

    assert all(output[0].numpy == arg1)
    assert all(output[1].numpy == arg2)
    assert all(output[2].numpy == arg3)

def test_ready_argument_list2():
    arg1 = numpy.array([1, 2, 3]).astype(numpy.float32)
    arg2 = numpy.int32(7)
    arg3 = numpy.float32(6.0)
    arguments = [arg1, arg2, arg3]

    cfunc = CFunctions()
    output = cfunc.ready_argument_list(arguments)
    print(output)

    output_arg1 = numpy.ctypeslib.as_array(output[0].ctypes, shape=arg1.shape)

    assert output_arg1.dtype == 'float32'
    assert isinstance(output[1].ctypes, C.c_int32)
    assert isinstance(output[2].ctypes, C.c_float)

    assert all(output_arg1 == arg1)
    assert output[1][1].value == arg2
    assert output[2][1].value == arg3

def test_ready_argument_list3():
    arg1 = Mock()
    arguments = [arg1]
    cfunc = CFunctions()
    try:
        cfunc.ready_argument_list(arguments)
        assert False
    except Exception:
        assert True

def test_ready_argument_list4():
    with raises(TypeError):
        arg1 = int(9)
        cfunc = CFunctions()
        cfunc.ready_argument_list([arg1])

def test_ready_argument_list5():
    arg1 = numpy.array([1, 2, 3]).astype(numpy.float32)
    arguments = [arg1]

    cfunc = CFunctions()
    output = cfunc.ready_argument_list(arguments)

    assert all(output[0].numpy == arg1)

    # test that a copy has been made
    arg1[0] = arg1[0] + 1
    assert not all(output[0].numpy == arg1)


def test_byte_array_arguments():
    arg1 = numpy.array([1, 2, 3]).astype(numpy.int8)

    cfunc = CFunctions()

    output = cfunc.ready_argument_list([arg1])

    output_arg1 = numpy.ctypeslib.as_array(output[0].ctypes, shape=arg1.shape)

    assert output_arg1.dtype == 'int8'

    assert all(output_arg1 == arg1)

    dest = numpy.zeros_like(arg1)

    cfunc.memcpy_dtoh(dest, output[0])

    assert all(dest == arg1)


@patch('kernel_tuner.c.subprocess')
@patch('kernel_tuner.c.numpy.ctypeslib')
def test_compile(npct, subprocess):

    kernel_string = "this is a fake C program"
    kernel_name = "blabla"
    kernel_sources = KernelSource(kernel_string, "C")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])

    cfunc = CFunctions()

    f = cfunc.compile(kernel_instance)

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


@patch('kernel_tuner.c.subprocess')
@patch('kernel_tuner.c.numpy.ctypeslib')
def test_compile_detects_device_code(npct, subprocess):

    kernel_string = "this code clearly contains device code __global__ kernel(float* arg){ return; }"
    kernel_name = "blabla"
    kernel_sources = KernelSource(kernel_string, "C")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])

    cfunc = CFunctions()

    f = cfunc.compile(kernel_instance)

    print(subprocess.check_call.call_args_list)

    #assert the filename suffix used for source compilation is .cu
    dot_cu_used = False
    for call in subprocess.check_call.call_args_list:
        args, kwargs = call
        args = args[0]
        print(args)
        if args[0] == 'nvcc' and args[1] == '-c':
            assert args[2][-3:] == '.cu'
            dot_cu_used = True

    assert dot_cu_used



def test_memset():
    a = [1, 2, 3, 4]
    x = numpy.array(a).astype(numpy.float32)
    x_c = x.ctypes.data_as(C.POINTER(C.c_float))
    arg = Argument(numpy=x, ctypes=x_c)

    cfunc = CFunctions()
    cfunc.memset(arg, 0, x.nbytes)

    output = numpy.ctypeslib.as_array(x_c, shape=(4,))

    print(output)
    assert all(output == numpy.zeros(4))
    assert all(x == numpy.zeros(4))

def test_memcpy_dtoh():
    a = [1, 2, 3, 4]
    x = numpy.array(a).astype(numpy.float32)
    x_c = x.ctypes.data_as(C.POINTER(C.c_float))
    arg = Argument(numpy=x, ctypes=x_c)
    output = numpy.zeros_like(x)

    cfunc = CFunctions()
    cfunc.memcpy_dtoh(output, arg)

    print(a)
    print(output)

    assert all(output == a)
    assert all(x == a)

def test_memcpy_htod():
    a = [1, 2, 3, 4]
    src = numpy.array(a).astype(numpy.float32)
    x = numpy.zeros_like(src)
    x_c = x.ctypes.data_as(C.POINTER(C.c_float))
    arg = Argument(numpy=x, ctypes=x_c)

    cfunc = CFunctions()
    cfunc.memcpy_htod(arg, src)

    assert all(arg.numpy == a)
