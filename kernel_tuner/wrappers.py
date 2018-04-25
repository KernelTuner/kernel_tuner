""" Module for wrapper functions

This module contains functions that generate wrappers for functions,
allowing them to be compiled and run using Kernel Tuner.

The first function in this module generates a wrapper for
primitive-typed (templated) C++ functions, allowing them to be
compiled and executed using Kernel Tuner. The plan is to later add
functionality to also wrap device functions.

"""

import numpy as np

from kernel_tuner import util


def cpp(function_name, kernel_source, args, convert_to_array=None):
    """ Generate a wrapper to call C++ functions from Python

    This function allows Kernel Tuner to call templated C++ functions
    that use primitive data types (double, float, int, ...).

    There is support to convert function arguments from plain pointers
    to array references. If this is needed, there should be a True value
    in convert_to_array in the location corresponding to the location in
    the args array.

    For example, a Numpy array argument of type float64 and length 10
    will be cast using:
    ``*reinterpret_cast<double(*)[10]>(arg)``
    which allows it to be used to call a C++ that is defined as:
    ``template<typename T, int s>void my_function(T (&arg)[s], ...)``

    Arrays of size 1 will be converted to simple non-array references.
    False indicates that no conversion is performed. Conversion
    is only support for numpy array arguments. If convert_to_array is
    passed it should have the same length as the args array.

    :param function_name: A string containing the name of the C++ function
        to be wrapped
    :type function_name: string

    :param kernel_source: One of the sources for the kernel, could be a
        function that generates the kernel code, a string containing a filename
        that points to the kernel source, or just a string that contains the code.
    :type kernel_source: string or callable

    :param args: A list of kernel arguments, use numpy arrays for
        arrays, use numpy.int32 or numpy.float32 for scalars.
    :type args: list

    :param convert_to_array: A list of same length as args, containing
        True or False values indicating whether the corresponding argument
        in args should be cast to a reference to an array or not.
    :type convert_to_array: list (True or False)

    :returns: A string containing the orignal code extended with the wrapper
        function. The wrapper has "extern C" binding and can be passed to
        other Kernel Tuner functions, for example run_kernel with lang="C".
        The name of the wrapper function will be the name of the function with
        a "_wrapper" postfix.
    :rtype: string

    """

    if convert_to_array and len(args) != len(convert_to_array):
        raise ValueError("convert_to_array length should be same as args")

    type_map = {"int8": "char",
                "int16": "short",
                "int32": "int",
                "float32": "float",
                "float64": "double"}

    def type_str(arg):
        if not str(arg.dtype) in type_map:
            raise Value("only primitive data types are supported by the C++ wrapper")
        typestring = type_map[str(arg.dtype)]
        if isinstance(arg, np.ndarray):
            typestring += " *"
        return typestring + " "

    signature = ",".join([type_str(arg) + "arg" + str(i) for i, arg in enumerate(args)])

    if not convert_to_array:
        call_args = ",".join(["arg" + str(i) for i in range(len(args))])
    else:
        call_args = []
        for i, arg in enumerate(args):
            if convert_to_array[i]:
                if not isinstance(arg, np.ndarray):
                    ValueError("conversion to array reference only supported for arguments that are numpy arrays, use length-1 numpy array to pass a scalar by reference")
                if np.prod(arg.shape) > 1:
                    #convert pointer to a reference to an array
                    arg_shape = "".join("[%d]" % i for i in arg.shape)
                    arg_str = "*reinterpret_cast<" + type_map[str(arg.dtype)] + "(*)" + arg_shape + ">(arg" + str(i) + ")"
                else:
                    #a reference is accepted rather than a pointer, just dereference
                    arg_str = "*arg" + str(i)
                call_args.append(arg_str)
                #call_args = ",".join(["*reinterpret_cast<double(*)[9]>(arg" + str(i) + ")" for i in range(len(args))])
            else:
                call_args.append("arg" + str(i))
        call_args_str = ",".join(call_args)

    kernel_string = util.get_kernel_string(kernel_source)

    return """

    %s

    extern "C"
    float %s_wrapper(%s) {

        %s(%s);

        return 0.0f;
    }""" % (kernel_string, function_name, signature, function_name, call_args_str)


