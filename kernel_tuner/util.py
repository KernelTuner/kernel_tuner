""" Module for kernel tuner utility functions """
from __future__ import print_function
import numpy
import os

def get_temp_filename():
    random_large_int = numpy.random.randint(low=1000000, high=1000000000)
    return 'temp_' + str(random_large_int)

def looks_like_a_filename(original_kernel):
    """ attempt to detect whether source code or a filename was passed """
    result = False
    if isinstance(original_kernel, str):
        result = True
        #test if not too long
        if len(original_kernel) > 100:
            result = False
        #test if not contains special characters
        for c in "();{}\\":
            if c in original_kernel:
                result = False
        #just a safeguard for stuff that looks like code
        for s in ["__global__ ", "__kernel ", "void ", "float "]:
            if s in original_kernel:
                result = False
        #string must contain substring ".c"
        result = result and ".c" in original_kernel
    return result

def read_file(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            return f.read()
    return None

def detect_language(lang, original_kernel):
    """attempt to detect language from the kernel_string if not specified"""
    if lang is None:
        kernel_string = original_kernel
        if looks_like_a_filename(original_kernel):
            kernel_string = read_file(original_kernel) or original_kernel
        if "__global__" in kernel_string:
            lang = "CUDA"
        elif "__kernel" in kernel_string:
            lang = "OpenCL"
        else:
            lang = "C"
    return lang

def get_grid_dimensions(problem_size, params, grid_div_y, grid_div_x):
    """compute grid dims based on problem sizes and listed grid divisors"""
    current_problem_size = []
    for s in problem_size:
        if isinstance(s, str):
            current_problem_size.append(int(eval(replace_param_occurrences(s,params))))
        elif isinstance(s, int) or isinstance(s, numpy.integer):
            current_problem_size.append(s)
        else:
            raise TypeError("Error: problem_size should only be list of string or int")
    div_x = 1
    if grid_div_x is None and "block_size_x" in params:
        grid_div_x = ["block_size_x"]
    if grid_div_x is not None:
        div_x = numpy.prod([int(eval(replace_param_occurrences(s,params))) for s in grid_div_x])
    div_y = 1
    if grid_div_y is not None:
        div_y = numpy.prod([int(eval(replace_param_occurrences(s,params))) for s in grid_div_y])
    grid = (int(numpy.ceil(float(current_problem_size[0]) / float(div_x))),
            int(numpy.ceil(float(current_problem_size[1]) / float(div_y))) )
    return grid

def get_thread_block_dimensions(params):
    """thread block size from tuning params, currently using convention"""
    block_size_x = params.get("block_size_x", 256)
    block_size_y = params.get("block_size_y", 1)
    block_size_z = params.get("block_size_z", 1)
    return (block_size_x, block_size_y, block_size_z)

def prepare_kernel_string(original_kernel, params, grid=(1,1)):
    """prepend the kernel with a series of C preprocessor defines"""
    kernel_string = original_kernel
    kernel_string = "#define grid_size_x " + str(grid[0]) + "\n" + kernel_string
    kernel_string = "#define grid_size_y " + str(grid[1]) + "\n" + kernel_string
    for k, v in params.items():
        kernel_string = "#define " + k + " " + str(v) + "\n" + kernel_string
    return kernel_string

def replace_param_occurrences(string, params):
    """replace occurrences of the tuning params with their current value"""
    for k, v in params.items():
        string = string.replace(k, str(v))
    return string

def check_restrictions(restrictions, element, keys, verbose):
    params = dict(zip(keys, element))
    for restrict in restrictions:
        if not eval(replace_param_occurrences(restrict, params)):
            if verbose:
                instance_string = "_".join([str(i) for i in element])
                print("skipping config", instance_string, "reason: config fails restriction")
            return False
    return True

def check_argument_list(args):
    for (i, arg) in enumerate(args):
        if not isinstance(arg, (numpy.ndarray, numpy.generic)):
            raise TypeError("Argument at position " + str(i) + " of type: " + str(type(arg)) + " should be of type numpy.ndarray or numpy scalar")

def setup_block_and_grid(dev, problem_size, grid_div_y, grid_div_x, params, instance_string, verbose):
        """compute thread block and grid dimensions for this kernel"""
        threads = get_thread_block_dimensions(params)
        if numpy.prod(threads) > dev.max_threads:
            if verbose:
                print("skipping config", instance_string, "reason: too many threads per block")
            return None, None
        grid = get_grid_dimensions(problem_size, params, grid_div_y, grid_div_x)
        return threads, grid

def setup_kernel_strings(kernel_name, original_kernel, params, grid, instance_string):
        """create configuration specific kernel string"""
        kernel_string = original_kernel
        if looks_like_a_filename(original_kernel):
            kernel_string = read_file(original_kernel) or original_kernel

        kernel_string = prepare_kernel_string(kernel_string, params, grid)
        name = kernel_name + "_" + instance_string
        kernel_string = kernel_string.replace(kernel_name, name)
        return name, kernel_string

