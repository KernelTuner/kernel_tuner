""" 
A simple CUDA/OpenCL kernel tuner in Python
===========================================

The goal of this project is to provide a - as simple as possible - tool
for tuning CUDA and OpenCL kernels. This implies that any CUDA or OpenCL
kernel can be tuned without requiring extensive changes to the original
kernel code.

A very common problem in GPU programming is that some combination of
thread block dimensions and other kernel parameters, like tiling or
unrolling factors, results in dramatically better performance than other
kernel configurations. The goal of auto-tuning is to automate the
process of finding the best performing configuration for a given device.

This kernel tuner aims that you can directly use the tuned kernel
without introducing any new dependencies. The tuned kernels can
afterwards be used independently of the programming environment, whether
that is using C/C++/Java/Fortran or Python doesn't matter.

The kernel_tuner module currently only contains main one function which
is called tune_kernel to which you pass at least the kernel name, a string
containing the kernel code, the problem size, a list of kernel function
arguments, and a dictionary of tunable parameters. There are also a lot
of optional parameters, for a complete list see the full documentation of
tune_kernel.

Installation
------------
| clone the repository
|  ``git clone git@github.com:benvanwerkhoven/kernel_tuner.git``
| change into the top-level directory
|  ``cd kernel_tuner``
| install using
|  ``pip install -r requirements.txt``
|  ``pip install .``

Dependencies
------------
 * Python 2.7 or Python 3.5
 * PyCuda and/or PyOpenCL (https://mathema.tician.de/software/)

Example usage
-------------
The following shows a simple example for tuning a CUDA kernel:

::

    kernel_string = \"\"\"
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    \"\"\"

    size = 10000000
    problem_size = (size, 1)

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)
    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    tune_kernel("vector_add", kernel_string, problem_size, args, tune_params)

The exact same Python code can be used to tune an OpenCL kernel:

::

    kernel_string = \"\"\"
    __kernel void vector_add(__global float *c, __global float *a, __global float *b, int n) {
        int i = get_global_id(0);
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    \"\"\"


Or even just a C function, see the example `here <https://github.com/benvanwerkhoven/kernel_tuner/blob/master/examples/c/vector_add.py>`_.

You can find these and many - more extensive - example codes, in the
`examples <https://github.com/benvanwerkhoven/kernel_tuner/blob/master/examples/>`_
directory. See the `full documentation <http://benvanwerkhoven.github.io/kernel_tuner/sphinxdoc/html/index.html>`_
for several highly detailed tutorial-style explanations of example
kernels and the scripts to tune them.

Author
------
Ben van Werkhoven <b.vanwerkhoven@esciencenter.nl>

Copyright and License
---------------------
* Copyright 2016 Netherlands eScience Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function

import numpy
import itertools

from kernel_tuner.util import *
from kernel_tuner.core import get_device_interface

def tune_kernel(kernel_name, kernel_string, problem_size, arguments,
        tune_params, grid_div_x=None, grid_div_y=None,
        restrictions=None, answer=None, atol=1e-6, verbose=False,
        lang=None, device=0, platform=0, cmem_args=None,
        num_threads=1, use_noodles=False, sample=False, compiler_options=None):
    """ Tune a CUDA kernel given a set of tunable parameters

    :param kernel_name: The name of the kernel in the code.
    :type kernel_name: string

    :param kernel_string: The CUDA, OpenCL, or C kernel code as a string.
            It is also allowed for the string to be a filename of the file
            containing the code.

            To support combined host and device code tuning for runtime
            compiled device code, a list of filenames can be passed instead.
            The first file in the list should be the file that contains the
            host code. The host code is allowed to include or read as a string
            any of the files in the list beyond the first.
    :type kernel_string: string or list

    :param problem_size: A tuple containing the size from which the grid
            dimensions of the kernel will be computed. Do not divide by
            the thread block sizes, if this is necessary use grid_div_x/y to
            specify.

            If you want you can use a string to specify a problem
            size, within these strings you are allowed to write Python
            arithmetic and use the tuning parameter names as variables.
            The kernel tuner will replace instances of the tuning parameter
            names with their current value while iterating over the search
            space. See the reduction CUDA example for an example use of this
            feature.
    :type problem_size: tuple(int or string, int or string)

    :param arguments: A list of kernel arguments, use numpy arrays for
            arrays, use numpy.int32 or numpy.float32 for scalars.
    :type arguments: list

    :param tune_params: A dictionary containing the parameter names as keys,
            and lists of possible parameter settings as values.
            The kernel tuner will try to compile and benchmark all possible
            combinations of all possible values for all tuning parameters.
            This typically results in a rather large search space of all
            possible kernel configurations.
            For each kernel configuration, each tuning parameter is
            replaced at compile-time with its current value.
            Currently, the kernel tuner uses the convention that the following
            list of tuning parameters are used as thread block dimensions:

                * "block_size_x"   thread block (work group) x-dimension
                * "block_size_y"   thread block (work group) y-dimension
                * "block_size_z"   thread block (work group) z-dimension

            Options for changing these defaults may be added later. If you
            don't want the thread block dimensions to be compiled in, you
            may use the built-in variables blockDim.xyz in CUDA or the
            built-in function get_local_size() in OpenCL instead.
    :type tune_params: dict( string : [...] )

    :param grid_div_x: A list of names of the parameters whose values divide
        the grid dimensions in the x-direction. Arithmetic expressions can be
        used if necessary inside the string containing a parameter name. For
        example, in some cases you may want to divide the problem size in the
        x-dimension with the number of warps rather than the number of threads
        in a block, in such cases one could use ["block_size_x/32"]. Note that
        the product of all grid divisor expressions is computed before dividing
        the problem_size in that dimension. Also note that the divison is treated
        as a float divison and resulting grid dimensions will be rounded up to
        the nearest integer number.
        If not supplied, ["block_size_x"] will be used by default, if you do not
        want any grid x-dimension divisors pass an empty list.
    :type grid_div_x: list

    :param grid_div_y: A list of names of the parameters whose values divide
        the grid dimensions in the y-direction, None by default. See grid_div_x
        for more details.
    :type grid_div_y: list

    :param restrictions: A list of strings containing boolean expression that
        limit the search space in that they must be satisfied by the kernel
        configuration. These expressions must be true for the configuration
        to be part of the search space. For example:
        restrictions=["block_size_x==block_size_y*tile_size_y"] limits the
        search to configurations where the block_size_x equals the product
        of block_size_y and tile_size_y.
        The default is None.
    :type restrictions: list

    :param answer: A list of arguments, similar to what you pass to arguments,
        that contains the expected output of the kernel after it has executed
        and contains None for each argument that is input-only. The expected
        output of the kernel will then be used to verify the correctness of
        each kernel in the parameter space before it will be benchmarked.
    :type answer: list

    :param atol: The maximum allowed absolute difference between two elements
        in the output and the reference answer, as passed to numpy.allclose().
        Ignored if you have not passed a reference answer. Default value is
        1e-6, that is 0.000001.
    :type atol: float

    :param verbose: Sets whether or not to report about configurations that
        were skipped during the search. This could be due to several reasons:

            * kernel configuration fails one or more restrictions
            * too many threads per thread block
            * too much shared memory used by the kernel
            * too many resources requested for launch

        verbose is set to False by default.
    :type verbose: boolean

    :param lang: Specifies the language used for GPU kernels. The kernel_tuner
        automatically detects the language, but if it fails, you may specify
        the language using this argument, currently supported: "CUDA", "OpenCL", or "C"
    :type lang: string

    :param device: CUDA/OpenCL device to use, in case you have multiple
        CUDA-capable GPUs or OpenCL devices you may use this to select one,
        0 by default. Ignored if you are tuning host code by passing lang="C".
    :type device: int

    :param platform: OpenCL platform to use, in case you have multiple
        OpenCL platforms you may use this to select one,
        0 by default. Ignored if not using OpenCL.
    :type device: int

    :param cmem_args: CUDA-specific feature for specifying constant memory
        arguments to the kernel. In OpenCL these are handled as normal
        kernel arguments, but in CUDA you can copy to a symbol. The way you
        specify constant memory arguments is by passing a dictionary with
        strings containing the constant memory symbol name together with numpy
        objects in the same way as normal kernel arguments.
    :type cmem_args: dict(string: numpy object)

    :param compiler_options: A list of strings that specifies compiler options.
    :type compiler_options: list(string)

    :returns: A list of dictionaries of all executed kernel configurations and their
        execution times.
    :rtype: list(dict())
    """

    #see if the kernel arguments have correct type
    check_argument_list(arguments)

    #compute cartesian product of all tunable parameters
    parameter_space = itertools.product(*tune_params.values())

    #check for search space restrictions
    if restrictions is not None:
        parameter_space = filter(lambda p: check_restrictions(restrictions, p, tune_params.keys(), verbose), parameter_space)

    #if running sequential
    if sample == True:
        import kernel_tuner.runners.random_sample as runner
    elif num_threads == 1 and use_noodles == False:
        import kernel_tuner.runners.sequential_brute_force as runner
    else:
        raise NotImplementedError("parallel runners will be implemented soon")

    results = runner.run(kernel_name, kernel_string, problem_size, arguments,
        tune_params, parameter_space, grid_div_x, grid_div_y,
        answer, atol, verbose,
        lang, device, platform, cmem_args, compiler_options)

    #finished iterating over search space
    if len(results) > 0:
        best_config = min(results, key=lambda x:x['time'])
        print("best performing configuration:", "".join([k + "=" + str(v) + ", " for k,v in best_config.items()]))
    else:
        print("no results to report")

    return results



def run_kernel(kernel_name, kernel_string, problem_size, arguments,
        params, grid_div_x=None, grid_div_y=None,
        lang=None, device=0, platform=0, cmem_args=None, compiler_options=None):
    """Compile and run a single kernel

    Compiles and runs a single kernel once, given a specific instance of the kernels tuning parameters.
    This function was added to the kernel tuner mostly for verifying kernel correctness.
    On purpose, it is called much in the same way as `tune_kernel()`

    :param kernel_name: The name of the kernel in the code
    :type kernel_name: string

    :param kernel_string: The CUDA or OpenCL kernel code as a string
    :type kernel_string: string

    :param problem_size: A tuple containing the size from which the grid
            dimensions of the kernel will be computed. Do not divide by
            the thread block sizes, if this is necessary use grid_div_x/y to
            specify.
    :type problem_size: tuple(int, int)

    :param arguments: A list of kernel arguments, use numpy arrays for
            arrays, use numpy.int32 or numpy.float32 for singulars
    :type arguments: list

    :param params: A dictionary containing the tuning parameter names as keys
            and a single value per tuning parameter as values.
    :type params: dict( string: int )

    :param grid_div_x: See tune_kernel()
    :type grid_div_x: list

    :param grid_div_y: See tune_kernel()
    :type grid_div_y: list

    :param lang: Language of the kernel, supply "CUDA", "OpenCL", or "C" if not detected automatically.
    :type lang: string

    :param device: CUDA/OpenCL device to use, 0 by default.
    :type device: int

    :param platform: OpenCL platform to use, in case you have multiple
        OpenCL platforms you may use this to select one,
        0 by default. Ignored if not using OpenCL.
    :type device: int

    :param cmem_args: CUDA-specific feature for specifying constant memory
        arguments to the kernel. See tune_kernel() for details.
    :type cmem_args: dict(string, ...)

    :param compiler_options: A list of strings that specifies compiler options.
    :type compiler_options: list(string)

    :returns: A list of numpy arrays, similar to the arguments passed to this
        function, containing the output after kernel execution.
    :rtype: list
    """

    #move data to the GPU and compile the kernel
    lang = detect_language(lang, kernel_string)
    dev = get_device_interface(lang, device, platform, compiler_options)
    check_argument_list(arguments)
    gpu_args = dev.ready_argument_list(arguments)

    #retrieve the run configuration, compile, and run the kernel
    threads = get_thread_block_dimensions(params)
    grid = get_grid_dimensions(problem_size, params,
                       grid_div_y, grid_div_x)

    kernel_string = prepare_kernel_string(kernel_string, params, grid)
    func = dev.compile(kernel_name, kernel_string)

    #add constant memory arguments to compiled module
    if cmem_args is not None:
        dev.copy_constant_memory_args(cmem_args)

    dev.run_kernel(func, gpu_args, threads, grid)

    #copy data in GPU memory back to the host
    results = []
    for i, arg in enumerate(arguments):
        results.append(numpy.zeros_like(arg))
        dev.memcpy_dtoh(results[-1], gpu_args[i])
    return results


