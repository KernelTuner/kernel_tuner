""" Module for grouping the core functionality needed by most runners """
from __future__ import print_function

from kernel_tuner.cuda import CudaFunctions
from kernel_tuner.opencl import OpenCLFunctions
from kernel_tuner.c import CFunctions
from kernel_tuner.util import *

def get_device_interface(lang, device, platform, compiler_options=None):
    if lang == "CUDA":
        dev = CudaFunctions(device, compiler_options=compiler_options)
    elif lang == "OpenCL":
        dev = OpenCLFunctions(device, platform, compiler_options=compiler_options)
    elif lang == "C":
        dev = CFunctions(compiler_options=compiler_options)
    else:
        raise UnImplementedException("Sorry, support for languages other than CUDA, OpenCL, or C is not implemented yet")
    return dev

def check_kernel_correctness(dev, func, gpu_args, threads, grid, answer, params, verbose, atol=1e-6):
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
                print("Error: " + get_config_string(params) + " detected during correctness check")
                print("Printing kernel output and expected result, set verbose=False to suppress this debug print")
                numpy.set_printoptions(edgeitems=500)
                print("Kernel output:")
                print(result_host)
                print("Expected:")
                print(expected)
            correct = correct and output_test
    if not correct:
        raise Exception("Error: " + get_config_string(params) + " failed correctness check")
    return correct

def compile_kernel(dev, kernel_name, kernel_string, params, grid, instance_string, verbose):
    """compile the kernel for this specific instance"""

    #prepare kernel_string for compilation
    name, kernel_string = setup_kernel_strings(kernel_name, kernel_string, params, grid, instance_string)

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
            print("Error while compiling:", instance_string)
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

def compile_and_benchmark(dev, gpu_args, kernel_name, original_kernel, params,
        problem_size, grid_div_y, grid_div_x, cmem_args, answer, atol, instance_string, verbose):

    #setup thread block and grid dimensions
    threads, grid = setup_block_and_grid(dev, problem_size, grid_div_y, grid_div_x, params, instance_string, verbose)
    if threads is None:
        return None

    temp_files = dict()

    try:
        #obtain the kernel_string and prepare additional files, if any
        if isinstance(original_kernel, list):
            kernel_string, temp_files = prepare_list_of_files(original_kernel, params, grid)
        else:
            kernel_string = get_kernel_string(original_kernel)

        #compile the kernel
        func = compile_kernel(dev, kernel_name, kernel_string, params, grid, instance_string, verbose)
        if func is None:
            return None

        #add constant memory arguments to compiled module
        if cmem_args is not None:
            dev.copy_constant_memory_args(cmem_args)

        #test kernel for correctness and benchmark
        if answer is not None:
            check_kernel_correctness(dev, func, gpu_args, threads, grid, answer, params, verbose, atol)

        #benchmark
        time = benchmark(dev, func, gpu_args, threads, grid, instance_string, verbose)

    except Exception as e:
        _, kernel_string = setup_kernel_strings(kernel_name, kernel_string, params, grid, instance_string)
        print("Error while compiling or benchmarking the following code:\n" + kernel_string + "\n")
        raise e

    #clean up any temporary files
    finally:
        for v in temp_files.values():
            delete_temp_file(v)

    return time


