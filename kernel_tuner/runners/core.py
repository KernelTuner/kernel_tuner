from __future__ import print_function

import numpy
from kernel_tuner.util import *

def check_kernel_correctness(dev, func, gpu_args, threads, grid, answer, instance_string, atol=1e-6):
    """runs the kernel once and checks the result against answer"""
    for result, expected in zip(gpu_args, answer):
        if expected is not None:
            dev.memset(result, 0, expected.nbytes)
    dev.run_kernel(func, gpu_args, threads, grid)
    correct = True
    for result,expected in zip(gpu_args,answer):
        if expected is not None:
            result_host = numpy.zeros_like(expected)
            dev.memcpy_dtoh(result_host, result)
            correct = correct and numpy.allclose(result_host.ravel(), expected.ravel(), atol=atol)
    if not correct:
        raise Exception("Error " + instance_string + " failed correctness check")
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

def compile(dev, kernel_name, original_kernel, params, grid, instance_string, verbose):
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
