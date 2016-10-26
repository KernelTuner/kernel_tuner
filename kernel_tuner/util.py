""" Module for kernel tuner utility functions """
from __future__ import print_function
import numpy
import os
import errno

def get_instance_string(params):
    return "_".join([str(i) for i in params.values()])

def get_config_string(params):
    return "".join([k + "=" + str(v) + ", " for k,v in params.items()])


def get_kernel_string(original_kernel):
    kernel_string = original_kernel
    if looks_like_a_filename(original_kernel):
        kernel_string = read_file(original_kernel) or original_kernel
    return kernel_string

def delete_temp_file(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise e

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

def write_file(filename, string):
    #ugly fix, hopefully we can find a better one
    import sys
    if sys.version_info[0] >= 3:
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(string)
    else:
        with open(filename, 'w') as f:
            f.write(string.encode("utf-8"))

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


def prepare_list_of_files(kernel_file_list, params, grid):
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

    kernel_string = get_kernel_string(kernel_file_list[0])
    if len(kernel_file_list) > 1:
        for f in kernel_file_list[1:]:
            #generate temp filename with the same extension
            temp_file = get_temp_filename() + "." + f.split(".")[-1]
            temp_files[f] = temp_file
            #add preprocessor statements to the additional file
            temp_file_string = prepare_kernel_string(get_kernel_string(f) , params, grid)
            write_file(temp_file, temp_file_string)
            #replace occurences of the additional file's name in the kernel_string with the name of the temp file
            kernel_string = kernel_string.replace(f, temp_file)

    return kernel_string, temp_files


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
        kernel_string = prepare_kernel_string(original_kernel, params, grid)
        name = kernel_name + "_" + instance_string
        kernel_string = kernel_string.replace(kernel_name, name)
        return name, kernel_string

