""" The default runner for iterating through the parameter space """
from __future__ import print_function

import numpy
import itertools
from collections import OrderedDict

from kernel_tuner.util import *

def run(kernel_name, original_kernel, problem_size, arguments,
        tune_params, parameter_space, grid_div_x, grid_div_y,
        answer, atol, verbose,
        lang, device, platform, cmem_args):
    """ Iterate through the entire parameter space using a single Python process

    :param kernel_name: The name of the kernel in the code.
    :type kernel_name: string

    :param original_kernel: The CUDA, OpenCL, or C kernel code as a string.
    :type original_kernel: string

    :param problem_size: See kernel_tuner.tune_kernel
    :type problem_size: tuple(int or string, int or string)

    :param arguments: A list of kernel arguments, use numpy arrays for
            arrays, use numpy.int32 or numpy.float32 for scalars.
    :type arguments: list

    :param tune_params: See kernel_tuner.tune_kernel
    :type tune_params: dict( string : [int, int, ...] )

    :param parameter_space: A list of lists that contains the entire parameter space
            to be searched. Each list in the list represents a single combination
            of parameters, order is imported and it determined by the order in tune_params.
    :type parameter_space: list( list() )

    :param grid_div_x: See kernel_tuner.tune_kernel
    :type grid_div_x: list

    :param grid_div_y: See kernel_tuner.tune_kernel
    :type grid_div_y: list

    :param answer: See kernel_tuner.tune_kernel
    :type answer: list

    :param atol: See kernel_tuner.tune_kernel
    :type atol: float

    :param verbose: See kernel_tuner.tune_kernel
    :type verbose: boolean

    :param lang: See kernel_tuner.tune_kernel
    :type lang: string

    :param device: See kernel_tuner.tune_kernel
    :type device: int

    :param platform: See kernel_tuner.tune_kernel
    :type device: int

    :param cmem_args: See kernel_tuner.tune_kernel
    :type cmem_args: dict(string: numpy object)

    :returns: A dictionary of all executed kernel configurations and their
        execution times.
    :rtype: dict( string, float )
    """

    results = dict()

    #detect language and create device function interface
    lang = detect_language(lang, original_kernel)
    dev = get_device_interface(lang, device, platform)
    max_threads = dev.max_threads

    #move data to the GPU
    gpu_args = dev.ready_argument_list(arguments)

    #iterate over parameter space
    for element in parameter_space:
        params = OrderedDict(zip(tune_params.keys(), element))
        instance_string = "_".join([str(i) for i in params.values()])

        #compute thread block and grid dimensions for this kernel
        threads = get_thread_block_dimensions(params)
        if numpy.prod(threads) > max_threads:
            if verbose:
                print("skipping config", instance_string, "reason: too many threads per block")
            continue
        grid = get_grid_dimensions(problem_size, params,
                       grid_div_y, grid_div_x)

        #create configuration specific kernel string
        kernel_string = prepare_kernel_string(original_kernel, params, grid)

        #rename the kernel to guarantee that PyCuda compiles a new kernel
        name = kernel_name + "_" + instance_string
        kernel_string = kernel_string.replace(kernel_name, name)

        #compile kernel func
        try:
            func = dev.compile(name, kernel_string)
        except Exception as e:
            #compiles may fail because certain kernel configurations use too
            #much shared memory for example, the desired behavior is to simply
            #skip over this configuration and try the next one
            if "uses too much shared data" in str(e):
                if verbose:
                    print("skipping config", instance_string, "reason: too much shared memory used")
                continue
            else:
                raise e

        #add constant memory arguments to compiled module
        if cmem_args is not None:
            dev.copy_constant_memory_args(cmem_args)

        #test kernel for correctness and benchmark
        if answer is not None:
            _check_kernel_correctness(dev, func, gpu_args, threads, grid, answer, instance_string, atol)

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
                continue
            else:
                print("Error while benchmarking:", instance_string)
                raise e

        #print the result
        print("".join([k + "=" + str(v) + ", " for k,v in params.items()]) + kernel_name + " took: " + str(time) + " ms.")
        results[instance_string] = time

    return results


#module private functions

def _check_kernel_correctness(dev, func, gpu_args, threads, grid, answer, instance_string, atol=1e-6):
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


