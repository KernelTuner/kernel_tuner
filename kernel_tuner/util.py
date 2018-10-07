""" Module for kernel tuner utility functions """
from __future__ import print_function

from collections import OrderedDict
import os
import errno
import tempfile
import logging
import warnings
import re
import numpy

default_block_size_names = ["block_size_x", "block_size_y", "block_size_z"]


def check_argument_type(dtype, kernel_argument, i):
    """check if the numpy.dtype matches the type used in the code"""
    types_map = {"uint8": ["uchar", "unsigned char", "uint8_t"],
                 "int8": ["char", "int8_t"],
                 "uint16": ["ushort", "unsigned short", "uint16_t"],
                 "int16": ["short", "int16_t"],
                 "uint32": ["uint", "unsigned int", "uint32_t"],
                 "int32": ["int", "int32_t"],   #discrepancy between OpenCL and C here, long may be 32bits in C
                 "uint64": ["ulong", "unsigned long", "uint64_t"],
                 "int64": ["long", "int64_t"],
                 "float16": ["half"],
                 "float32": ["float"],
                 "float64": ["double"]}
    if dtype in types_map:
        return any([substr in kernel_argument for substr in types_map[dtype]])
    else:
        return False # unknown dtype. do not throw exception to still allow kernel to run.

def check_argument_list(kernel_name, kernel_string, args):
    """ raise an exception if a kernel arguments do not match host arguments """
    kernel_arguments = list()
    collected_errors = list()
    for iterator in re.finditer(kernel_name + "[ \n\t]*" + "\(", kernel_string):
        kernel_start = iterator.end()
        kernel_end = kernel_string.find(")", kernel_start)
        if kernel_start != 0:
            kernel_arguments.append(kernel_string[kernel_start:kernel_end].split(","))
    for arguments_set, arguments in enumerate(kernel_arguments):
        collected_errors.append(list())
        if len(arguments) != len(args):
            collected_errors[arguments_set].append("Kernel and host argument lists do not match in size.")
            continue
        for (i, arg) in enumerate(args):
            kernel_argument = arguments[i]

            if not isinstance(arg, (numpy.ndarray, numpy.generic)):
                raise TypeError("Argument at position " + str(i) + " of type: " + str(type(arg)) + " should be of type numpy.ndarray or numpy scalar")

            correct = True
            if isinstance(arg, numpy.ndarray) and not "*" in kernel_argument:
                correct = False  #array is passed to non-pointer kernel argument

            if correct and check_argument_type(str(arg.dtype), kernel_argument, i):
                continue

            collected_errors[arguments_set].append("Argument at position " + str(i) + " of dtype: " + str(arg.dtype) +
                                                   " does not match " + kernel_argument + ".")
        if not collected_errors[arguments_set]:
            # We assume that if there is a possible list of arguments that matches with the provided one
            # it is the right one
            return
    for errors in collected_errors:
        warnings.warn(errors[0], UserWarning)
        #raise TypeError(errors[0])

def check_tune_params_list(tune_params):
    """ raise an exception if a tune parameter has a forbidden name """
    forbidden_names = ("grid_size_x", "grid_size_y", "grid_size_z")
    forbidden_name_substr = ("time", "times")
    for name, param in tune_params.items():
        if name in forbidden_names:
            raise ValueError("Tune parameter " + name + " with value " + str(param) + " has a forbidden name!")
        for forbidden_substr in forbidden_name_substr:
            if forbidden_substr in name:
                raise ValueError("Tune parameter " + name + " with value " + str(param) + " has a forbidden name: not allowed to use " + forbidden_substr + " in tune parameter names!")

def check_block_size_names(block_size_names):
    if block_size_names is not None:
        #do some type checks for the user input
        if not isinstance(block_size_names, list):
            raise ValueError("block_size_names should be a list of strings!")
        if len(block_size_names) > 3:
            raise ValueError("block_size_names should not contain more than 3 names!")
        if not all([isinstance(name, "".__class__) for name in block_size_names]):
            raise ValueError("block_size_names should contain only strings!")
        #ensure there is always at least three names
        for i, name in enumerate(default_block_size_names):
            if len(block_size_names) < i+1:
                block_size_names.append(name)

def check_block_size_params_names_list(block_size_names, tune_params):
    if block_size_names is not None:
        for name in block_size_names:
            if name not in tune_params.keys():
                warnings.warn("Block size name " + name + " is not specified in the tunable parameters list!", UserWarning)
    else: #if default block size names are used
        if not any([k in default_block_size_names for k in tune_params.keys()]):
            warnings.warn("None of the tunable parameters specify thread block dimensions!", UserWarning)


def check_restrictions(restrictions, element, keys, verbose):
    """ check whether a specific instance meets the search space restrictions """
    params = OrderedDict(zip(keys, element))
    for restrict in restrictions:
        if not eval(replace_param_occurrences(restrict, params)):
            if verbose:
                print("skipping config", get_instance_string(params), "reason: config fails restriction")
            return False
    return True

def delete_temp_file(filename):
    """ delete a temporary file, don't complain if is no longer exists """
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise e

def detect_language(lang, kernel_source):
    """attempt to detect language from the kernel_string if not specified"""
    if lang is None:
        if callable(kernel_source):
            raise TypeError("Please specify language when using a code generator function")
        kernel_string = get_kernel_string(kernel_source)
        if "__global__" in kernel_string:
            lang = "CUDA"
        elif "__kernel" in kernel_string:
            lang = "OpenCL"
        else:
            lang = "C"
    return lang


def get_config_string(params, units=None):
    """ return a compact string representation of a dictionary """
    compact_str_items = []
    # first make a list of compact strings for each parameter
    for k, v in params.items():
        unit = ""
        if isinstance(units, dict): #check if not None not enough, units could be mocked which causes errors
            unit = units.get(k, "")
        compact_str_items.append(k + "=" + str(v) + unit)
    # and finally join them
    compact_str = ", ".join(compact_str_items)
    return compact_str


def get_grid_dimensions(current_problem_size, params, grid_div, block_size_names):
    """compute grid dims based on problem sizes and listed grid divisors"""
    def get_dimension_divisor(divisor_list, default, params):
        if divisor_list is None:
            if default in params:
                divisor_list = [default]
            else:
                return 1
        return numpy.prod([int(eval(replace_param_occurrences(s, params))) for s in divisor_list])
    divisors = [get_dimension_divisor(d, block_size_names[i], params) for i, d in enumerate(grid_div)]
    return tuple(int(numpy.ceil(float(current_problem_size[i]) / float(d))) for i, d in enumerate(divisors))

def get_instance_string(params):
    """ combine the parameters to a string mostly used for debug output
        use of OrderedDict is advised
    """
    return "_".join([str(i) for i in params.values()])

def get_kernel_string(kernel_source, params=None):
    """ retrieve the kernel source and return as a string

    This function processes the passed kernel_source argument, which could be
    a function, a string with a filename, or just a string with code already.

    If kernel_source is a function, the function is called with instance
    parameters in 'params' as the only argument.

    If kernel_source looks like filename, the file is read in, but if
    the file does not exist, it is assumed that the string is not a filename
    after all.

    :param kernel_source: One of the sources for the kernel, could be a
        function that generates the kernel code, a string containing a filename
        that points to the kernel source, or just a string that contains the code.
    :type kernel_source: string or callable

    :param params: Dictionary containing the tunable parameters for this specific
        kernel instance, only needed when kernel_source is a generator.
    :type param: dict

    :returns: A string containing the kernel code.
    :rtype: string
    """
    #logging.debug('get_kernel_string called with %s', str(kernel_source))
    logging.debug('get_kernel_string called')

    kernel_string = None
    if callable(kernel_source):
        kernel_string = kernel_source(params)
    elif isinstance(kernel_source, str):
        if looks_like_a_filename(kernel_source):
            kernel_string = read_file(kernel_source) or kernel_source
        else:
            kernel_string = kernel_source
    else:
        raise TypeError("Error kernel_source is not a string nor a callable function")
    return kernel_string

def get_problem_size(problem_size, params):
    """compute current problem size"""
    if isinstance(problem_size, (str, int, numpy.integer)):
        problem_size = (problem_size, )
    current_problem_size = [1, 1, 1]
    for i, s in enumerate(problem_size):
        if isinstance(s, str):
            current_problem_size[i] = int(eval(replace_param_occurrences(s, params)))
        elif isinstance(s, (int, numpy.integer)):
            current_problem_size[i] = s
        else:
            raise TypeError("Error: problem_size should only contain strings or integers")
    return current_problem_size

def get_temp_filename(suffix=None):
    """ return a string in the form of temp_X, where X is a large integer """
    file = tempfile.mkstemp(suffix=suffix or "", prefix="temp_", dir=os.getcwd()) # or "" for Python 2 compatibility
    os.close(file[0])
    return file[1]

def get_thread_block_dimensions(params, block_size_names=None):
    """thread block size from tuning params, currently using convention"""
    if not block_size_names:
        block_size_names = default_block_size_names

    block_size_x = params.get(block_size_names[0], 256)
    block_size_y = params.get(block_size_names[1], 1)
    block_size_z = params.get(block_size_names[2], 1)
    return (int(block_size_x), int(block_size_y), int(block_size_z))

def looks_like_a_filename(kernel_source):
    """ attempt to detect whether source code or a filename was passed """
    result = False
    if isinstance(kernel_source, str):
        result = True
        #test if not too long
        if len(kernel_source) > 250:
            result = False
        #test if not contains special characters
        for c in "();{}\\":
            if c in kernel_source:
                result = False
        #just a safeguard for stuff that looks like code
        for s in ["__global__ ", "__kernel ", "void ", "float "]:
            if s in kernel_source:
                result = False
        #string must contain substring ".c", ".opencl", or ".F"
        result = result and any([s in kernel_source for s in (".c", ".opencl", ".F")])
    return result

def prepare_kernel_string(kernel_name, kernel_string, params, grid, threads, block_size_names):
    """ prepare kernel string for compilation

    Prepends the kernel with a series of C preprocessor defines specific
    to this kernel instance:

     * the thread block dimensions
     * the grid dimensions
     * tunable parameters

    Additionally the name of kernel is replace with an instance specific name. This
    is done to prevent that the kernel compilation could be skipped by PyCUDA and/or PyOpenCL,
    which may use caching to save compilation time. This feature could lead to strange bugs
    in the source code if the name of the kernel is also used for other stuff.

    :param kernel_name: Name of the kernel.
    :type kernel_name: string

    :param kernel_string: One of the source files of the kernel as a string containing code.
    :type kernel_string: string

    :param params: A dictionary containing the tunable parameters specific to this instance.
    :type params: dict

    :param grid: A tuple with the grid dimensions for this specific instance.
    :type grid: tuple(x,y,z)

    :param threads: A tuple with the thread block dimensions for this specific instance.
    :type threads: tuple(x,y,z)

    :param block_size_names: A tuple with the names of the thread block dimensions used
        in the code. By default this is ["block_size_x", ...], but the user
        may supply different names if they prefer.
    :type block_size_names: tuple(string)

    :returns: A string containing the source code made specific to this kernel instance.
    :rtype: string

    """
    logging.debug('prepare_kernel_string called for %s', kernel_name)

    grid_dim_names = ["grid_size_x", "grid_size_y", "grid_size_z"]
    for i, g in enumerate(grid):
        kernel_string = "#define " + grid_dim_names[i] + " " + str(g) + "\n" + kernel_string
    for i, g in enumerate(threads):
        kernel_string = "#define " + block_size_names[i] + " " + str(g) + "\n" + kernel_string
    for k, v in params.items():
        if k not in block_size_names:
            kernel_string = "#define " + k + " " + str(v) + "\n" + kernel_string
    name = kernel_name
    #name = kernel_name + "_" + get_instance_string(params)
    #kernel_string = kernel_string.replace(kernel_name, name)
    return name, kernel_string

def prepare_list_of_files(kernel_name, kernel_file_list, params, grid, threads, block_size_names):
    """ prepare the kernel string along with any additional files

    The first file in the list is allowed to include or read in the others
    The files beyond the first are considered additional files that may also contain tunable parameters

    For each file beyond the first this function creates a temporary file with
    preprocessors statements inserted. Occurences of the original filenames in the
    first file are replaced with their temporary counterparts.

    :param kernel_file_list: A list of filenames. The first file in the list is
        allowed to read or include the other files in the list. All files may
        will have access to the tunable parameters.
    :type kernel_file_list: list(string)

    :param params: A dictionary with the tunable parameters for this particular
        instance.
    :type params: dict()

    :param grid: The grid dimensions for this instance. The grid dimensions are
        also inserted into the code as if they are tunable parameters for
        convenience.
    :type grid: tuple()

    """
    temp_files = dict()

    kernel_string = get_kernel_string(kernel_file_list[0], params)
    name, kernel_string = prepare_kernel_string(kernel_name, kernel_string, params, grid, threads, block_size_names)

    if len(kernel_file_list) > 1:
        for f in kernel_file_list[1:]:
            #generate temp filename with the same extension
            temp_file = get_temp_filename(suffix="." + f.split(".")[-1])
            temp_files[f] = temp_file
            #add preprocessor statements to the additional file
            _, temp_file_string = prepare_kernel_string(kernel_name, get_kernel_string(f, params), params, grid, threads, block_size_names)
            write_file(temp_file, temp_file_string)
            #replace occurences of the additional file's name in the first kernel_string with the name of the temp file
            kernel_string = kernel_string.replace(f, temp_file)

    return name, kernel_string, temp_files

def read_file(filename):
    """ return the contents of the file named filename or None if file not found """
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            return f.read()

def replace_param_occurrences(string, params):
    """replace occurrences of the tuning params with their current value"""
    for k, v in params.items():
        string = string.replace(k, str(v))
    return string

def setup_block_and_grid(problem_size, grid_div, params, block_size_names=None):
    """compute problem size, thread block and grid dimensions for this kernel"""
    threads = get_thread_block_dimensions(params, block_size_names)
    current_problem_size = get_problem_size(problem_size, params)
    grid = get_grid_dimensions(current_problem_size, params, grid_div, block_size_names)
    return threads, grid

def write_file(filename, string):
    """dump the contents of string to a file called filename"""
    import sys
    #ugly fix, hopefully we can find a better one
    if sys.version_info[0] >= 3:
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(string)
    else:
        with open(filename, 'w') as f:
            f.write(string.encode("utf-8"))
