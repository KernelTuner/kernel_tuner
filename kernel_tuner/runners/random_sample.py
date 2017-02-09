""" Runner for taking a random sample of the parameter space """
from __future__ import print_function

import numpy
from collections import OrderedDict

from kernel_tuner.util import detect_language, get_instance_string, get_config_string
from kernel_tuner.core import get_device_interface, compile_and_benchmark

def run(kernel_name, original_kernel, problem_size, arguments,
        tune_params, parameter_space, grid_div_x, grid_div_y,
        answer, atol, verbose,
        lang, device, platform, cmem_args, compiler_options=None):
    """ Sample the parameter space using a single Python process

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

    results = []

    #detect language and create device function interface
    lang = detect_language(lang, original_kernel)
    dev = get_device_interface(lang, device, platform, compiler_options)

    #move data to the GPU
    gpu_args = dev.ready_argument_list(arguments)

    #reduce parameter space to a random sample of size 10%
    parameter_space = list(parameter_space)
    size = len(parameter_space)

    sample_indices = numpy.random.choice(range(size), size=int(numpy.ceil(size/float(10))), replace=False)
    print(kernel_name)

    #iterate over parameter space
    for i in sample_indices:
        element = parameter_space[i]

        params = OrderedDict(zip(tune_params.keys(), element))
        instance_string = get_instance_string(params)

        time = compile_and_benchmark(dev, gpu_args, kernel_name, original_kernel, params,
                        problem_size, grid_div_y, grid_div_x,
                        cmem_args, answer, atol, instance_string, verbose)
        if time is None:
            continue

        #print and append to results
        params['time'] = time
        print("".join([k + "=" + str(v) + ", " for k,v in params.items()]))
        results.append(params)

    return results



