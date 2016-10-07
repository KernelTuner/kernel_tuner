""" Module for kernel tuner utility functions """
from __future__ import print_function
import numpy

from kernel_tuner.cuda import CudaFunctions
from kernel_tuner.opencl import OpenCLFunctions
from kernel_tuner.c import CFunctions

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

def detect_language(lang, original_kernel):
    """attempt to detect language from the kernel_string if not specified"""
    kernel_string = original_kernel

    if looks_like_a_filename(original_kernel):
        with open(original_kernel, 'r') as f:
            kernel_string = f.read()

    if lang is None:
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

def check_kernel_correctness(dev, func, gpu_args, threads, grid, answer, instance_string, verbose, atol=1e-6):
    """runs the kernel once and checks the result against answer"""
    for result, expected in zip(gpu_args, answer):
        if expected is not None:
            dev.memset(result, 0, expected.nbytes)
    try:
        dev.run_kernel(func, gpu_args, threads, grid)
    except Exception as e:
        if "too many resources requested for launch" in str(e) or "OUT_OF_RESOURCES" in str(e):
            #ignore this error for now, it will show up when benchmarking the kernel
            return True
        else:
            raise e
    correct = True
    for result,expected in zip(gpu_args,answer):
        if expected is not None:
            result_host = numpy.zeros_like(expected)
            dev.memcpy_dtoh(result_host, result)
            output_test = numpy.allclose(result_host.ravel(), expected.ravel(), atol=atol)
            if not output_test and verbose:
                print("Error: " + instance_string + " detected during correctness check")
                print("Printing kernel output and expected result, set verbose=False to suppress this debug print")
                numpy.set_printoptions(edgeitems=500)
                print("Kernel output:")
                print(result_host)
                print("Expected:")
                print(expected)
            correct = correct and output_test
    if not correct:
        raise Exception("Error: " + instance_string + " failed correctness check")
    return correct

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

def compile_kernel(dev, kernel_name, original_kernel, params, grid, instance_string, verbose):
        """compile the kernel for this specific instance"""

        #prepare kernel_string for compilation
        name, kernel_string = setup_kernel_strings(kernel_name, original_kernel, params, grid, instance_string)

        #compile kernel_string into device func
        func = None
        try:
            func = dev.compile(name, kernel_string)
        except Exception as e:
            #compiles may fail because certain kernel configurations use too
            #much shared memory for example, the desired behavior is to simply
            #skip over this configuration and try the next one
            if "uses too much shared data" in str(e):
                if verbose:
                    print("skipping config", instance_string, "reason: too much shared memory used")
            else:
                raise e
        return func

def benchmark(dev, func, gpu_args, threads, grid, instance_string, verbose):
        """benchmark the kernel instance"""
        time = None
        try:
            time = dev.benchmark(func, gpu_args, threads, grid)
        except Exception as e:
            #some launches may fail because too many registers are required
            #to run the kernel given the current thread block size
            #the desired behavior is to simply skip over this configuration
            #and proceed to try the next one
            if "too many resources requested for launch" in str(e) or "OUT_OF_RESOURCES" in str(e):
                if verbose:
                    print("skipping config", instance_string, "reason: too many resources requested for launch")
            else:
                print("Error while benchmarking:", instance_string)
                raise e
        return time


def compile_and_benchmark(dev, gpu_args, kernel_name, original_kernel, params, problem_size, grid_div_y, grid_div_x, cmem_args, answer, atol, instance_string, verbose):
    #setup thread block and grid dimensions
    threads, grid = setup_block_and_grid(dev, problem_size, grid_div_y, grid_div_x, params, instance_string, verbose)
    if threads is None:
        return None

    #compile the kernel
    func = compile_kernel(dev, kernel_name, original_kernel, params, grid, instance_string, verbose)
    if func is None:
        return None

    #add constant memory arguments to compiled module
    if cmem_args is not None:
        dev.copy_constant_memory_args(cmem_args)

    #test kernel for correctness and benchmark
    if answer is not None:
        check_kernel_correctness(dev, func, gpu_args, threads, grid, answer, instance_string, verbose, atol)

    #benchmark
    time = benchmark(dev, func, gpu_args, threads, grid, instance_string, verbose)
    return time

