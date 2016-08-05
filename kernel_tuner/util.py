""" Module for kernel tuner utility functions """
import numpy

from kernel_tuner.cuda import CudaFunctions
from kernel_tuner.opencl import OpenCLFunctions
from kernel_tuner.c import CFunctions

def detect_language(lang, original_kernel):
    """attempt to detect language from the kernel_string if not specified"""
    if lang is None:
        if "__global__" in original_kernel:
            lang = "CUDA"
        elif "__kernel" in original_kernel:
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

def check_restrictions(restrictions, params):
    if restrictions != None:
        for restrict in restrictions:
            if not eval(replace_param_occurrences(restrict, params)):
                raise Exception("config fails restriction")

def get_device_interface(lang, device, platform):
    if lang == "CUDA":
        dev = CudaFunctions(device)
    elif lang == "OpenCL":
        dev = OpenCLFunctions(device, platform)
    elif lang == "C":
        dev = CFunctions()
    else:
        raise UnImplementedException("Sorry, support for languages other than CUDA, OpenCL, or C is not implemented yet")
    return dev

def check_argument_list(args):
    for (i, arg) in enumerate(args):
        if not isinstance(arg, (numpy.ndarray, numpy.generic)):
            raise TypeError("Argument at position " + str(i) + " of type: " + str(type(arg)) + " should be of type numpy.ndarray or numpy scalar")

