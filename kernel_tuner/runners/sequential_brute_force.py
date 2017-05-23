""" The default runner for iterating through the parameter space """
from __future__ import print_function

from collections import OrderedDict
import logging

from kernel_tuner.util import detect_language, get_config_string
from kernel_tuner.core import compile_and_benchmark, get_device_interface

def run(kernel_name, original_kernel, problem_size, arguments,
        tune_params, parameter_space, grid_div,
        answer, atol, verbose,
        lang, device, platform, cmem_args, compiler_options=None):
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

    :param grid_div: See kernel_tuner.tune_kernel
    :type grid_div: tuple(list)

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
    logging.debug('sequential_brute_force runner started for ' + kernel_name)

    results = []

    #detect language and create device function interface
    lang = detect_language(lang, original_kernel)
    dev = get_device_interface(lang, device, platform, compiler_options)

    #move data to the GPU
    gpu_args = dev.ready_argument_list(arguments)

    print("Using: " + dev.name)
    print(kernel_name)

    #iterate over parameter space
    for element in parameter_space:
        params = OrderedDict(zip(tune_params.keys(), element))

        time = compile_and_benchmark(dev, gpu_args, kernel_name, original_kernel, params,
                                     problem_size, grid_div,
                                     cmem_args, answer, atol, verbose)
        if time is None:
            logging.debug('received time is None, kernel configuration was skipped silently due to compile or runtime failure')
            continue

        #print and append to results
        params['time'] = time
        output_string = get_config_string(params)
        logging.debug(output_string)
        print(output_string)
        results.append(params)

    return results, dev.get_environment()
