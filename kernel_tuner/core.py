""" Module for grouping the core functionality needed by most runners """
from __future__ import print_function

import resource
import logging
import numpy

from kernel_tuner.cuda import CudaFunctions
from kernel_tuner.opencl import OpenCLFunctions
from kernel_tuner.c import CFunctions
import kernel_tuner.util as util

def get_device_interface(lang, device, platform, compiler_options=None):
    """ Instantiate the device functions interface, based on language """
    if lang == "CUDA":
        dev = CudaFunctions(device, compiler_options=compiler_options)
    elif lang == "OpenCL":
        dev = OpenCLFunctions(device, platform, compiler_options=compiler_options)
    elif lang == "C":
        dev = CFunctions(compiler_options=compiler_options)
    else:
        raise Exception("Sorry, support for languages other than CUDA, OpenCL, or C is not implemented yet")
    return dev

def run_kernel(dev, func, gpu_args, instance):
    """ Run a compiled kernel instance on a device """
    threads = instance["threads"]
    grid = instance["grid"]
    try:
        dev.run_kernel(func, gpu_args, threads, grid)
    except Exception as e:
        if "too many resources requested for launch" in str(e) or "OUT_OF_RESOURCES" in str(e):
            logging.debug('ignoring runtime failure due to too many resources required')
            return False
        else:
            logging.debug('encountered unexpected runtime failure: ' + str(e))
            raise e
    return True

def check_kernel_correctness(dev, func, gpu_args, instance, answer, verbose, atol=1e-6):
    """runs the kernel once and checks the result against answer"""
    logging.debug('check_kernel_correctness')
    params = instance["params"]

    #zero GPU memory for output arguments
    for result, expected in zip(gpu_args, answer):
        if expected is not None:
            dev.memset(result, 0, expected.nbytes)

    #run the kernel
    if not run_kernel(dev, func, gpu_args, instance):
        return True #runtime failure occured that should be ignored, skip correctness check

    #check correctness of each output argument
    correct = True
    for result, expected in zip(gpu_args, answer):
        if expected is not None:
            result_host = numpy.zeros_like(expected)
            dev.memcpy_dtoh(result_host, result)
            output_test = numpy.allclose(result_host.ravel(), expected.ravel(), atol=atol)
            if not output_test and verbose:
                print("Error: " + util.get_config_string(params) + " detected during correctness check")
                print("Printing kernel output and expected result, set verbose=False to suppress this debug print")
                numpy.set_printoptions(edgeitems=50)
                print("Kernel output:")
                print(result_host)
                print("Expected:")
                print(expected)
            correct = correct and output_test
            del result_host
    if not correct:
        logging.debug('correctness check has found a correctness issue')
        raise Exception("Error: " + util.get_config_string(params) + " failed correctness check")
    return correct

def compile_kernel(dev, instance, verbose):
    """compile the kernel for this specific instance"""
    logging.debug('compile_kernel ' + instance["name"])

    #compile kernel_string into device func
    func = None
    try:
        func = dev.compile(instance["name"], instance["kernel_string"])
    except Exception as e:
        #compiles may fail because certain kernel configurations use too
        #much shared memory for example, the desired behavior is to simply
        #skip over this configuration and try the next one
        if "uses too much shared data" in str(e):
            logging.debug('compile_kernel failed due to kernel using too much shared memory')
            if verbose:
                print("skipping config", instance["name"], "reason: too much shared memory used")
        else:
            logging.debug('compile_kernel failed due to error: ' + str(e))
            print("Error while compiling:", instance["name"])
            raise e
    return func

def benchmark(dev, func, gpu_args, instance, verbose):
    """benchmark the kernel instance"""
    logging.debug('benchmark ' + instance["name"])
    threads = instance["threads"]
    grid = instance["grid"]
    logging.debug('thread block dimensions x,y,z=%d,%d,%d', *threads)
    logging.debug('grid dimensions x,y,z=%d,%d,%d', *grid)

    time = None
    try:
        time = dev.benchmark(func, gpu_args, threads, grid)
    except Exception as e:
        #some launches may fail because too many registers are required
        #to run the kernel given the current thread block size
        #the desired behavior is to simply skip over this configuration
        #and proceed to try the next one
        if "too many resources requested for launch" in str(e) or "OUT_OF_RESOURCES" in str(e):
            logging.debug('benchmark fails due to runtime failure too many resources required')
            if verbose:
                print("skipping config", instance["name"], "reason: too many resources requested for launch")
        else:
            logging.debug('benchmark encountered runtime failure: ' + str(e))
            print("Error while benchmarking:", instance["name"])
            raise e
    return time

def create_kernel_instance(dev, kernel_name, original_kernel, problem_size, grid_div, params, verbose):
    """create kernel instance from kernel strings, parameters, problem size, grid divisors, and so on"""
    instance_string = util.get_instance_string(params)

    #setup thread block and grid dimensions
    threads, grid = util.setup_block_and_grid(problem_size, grid_div, params)
    if numpy.prod(threads) > dev.max_threads:
        if verbose:
            print("skipping config", instance_string, "reason: too many threads per block")
        return None

    temp_files = dict()
    #obtain the kernel_string and prepare additional files, if any
    if isinstance(original_kernel, list):
        kernel_string, temp_files = util.prepare_list_of_files(original_kernel, params, grid)
    else:
        kernel_string = util.get_kernel_string(original_kernel)

    #prepare kernel_string for compilation
    name, kernel_string = util.setup_kernel_strings(kernel_name, kernel_string, params, grid)

    #collect everything we know about this instance in a dict and return it
    instance = dict()
    instance["name"] = name
    instance["kernel_string"] = kernel_string
    instance["temp_files"] = temp_files
    instance["threads"] = threads
    instance["grid"] = grid
    instance["params"] = params
    return instance


def compile_and_benchmark(dev, gpu_args, kernel_name, original_kernel, params,
                          problem_size, grid_div, cmem_args, answer, atol, verbose):
    """ Compile and benchmark a kernel instance based on kernel strings and parameters """

    instance_string = util.get_instance_string(params)

    logging.debug('compile_and_benchmark ' + instance_string)
    mem_usage = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1)
    logging.debug('Memory usage : %2.2f MB', mem_usage)

    instance = create_kernel_instance(dev, kernel_name, original_kernel, problem_size, grid_div, params, verbose)
    if instance is None:
        return None

    try:
        #compile the kernel
        func = compile_kernel(dev, instance, verbose)
        if func is None:
            return None

        #add constant memory arguments to compiled module
        if cmem_args is not None:
            dev.copy_constant_memory_args(cmem_args)

        #test kernel for correctness and benchmark
        if answer is not None:
            check_kernel_correctness(dev, func, gpu_args, instance, answer, verbose, atol)

        #benchmark
        time = benchmark(dev, func, gpu_args, instance, verbose)

    except Exception as e:
        #dump kernel_string to temp file
        temp_filename = util.get_temp_filename() + ".c"
        util.write_file(temp_filename, instance["kernel_string"])
        print("Error while compiling or benchmarking, see source files: " + temp_filename + " ".join(instance["temp_files"].values()))
        raise e

    #clean up any temporary files, if no error occured
    for v in instance["temp_files"].values():
        util.delete_temp_file(v)

    return time
