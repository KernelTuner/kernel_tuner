"""Kernel Tuner interface module

This module contains the main functions that the Kernel Tuner
offers to its users.

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

from collections import OrderedDict
import importlib
from datetime import datetime
import logging
import sys
import numpy

import kernel_tuner.util as util
import kernel_tuner.core as core

from kernel_tuner.strategies import brute_force, random_sample, diff_evo, minimize, basinhopping

class Options(OrderedDict):
    """read-only class for passing options around"""
    def __getattr__(self, name):
        if not name.startswith('_'):
            return self[name]
        return super(Options, self).__getattr__(name)
    def __deepcopy__(self, _):
        return self


_kernel_options = Options([
    ("kernel_name", ("""The name of the kernel in the code.""", "string")),
    ("kernel_string", ("""The CUDA, OpenCL, or C kernel code as a string.
            It is also allowed for the string to be a filename of the file
            containing the code.

            To support combined host and device code tuning for runtime
            compiled device code, a list of filenames can be passed instead.
            The first file in the list should be the file that contains the
            host code. The host code is allowed to include or read as a string
            any of the files in the list beyond the first.

            Another alternative is to pass a function instead, or instead
            of the first item in the list of filenames. The purpose of this
            is to support the use of code generating functions that generate
            the kernel code based on the specific parameters. This function
            should take one positional argument, which will be used to pass
            a dict containing the parameters. The function should return a
            string with the source code for the kernel.""",
            "string or list and/or callable")),
    ("problem_size", ("""An int or string, or 1,2,3-dimensional tuple
            containing the size from which the grid dimensions of the kernel
            will be computed.

            Do not divide the problem_size yourself by the thread block sizes.
            The Kernel Tuner does this for you based on tunable parameters,
            called "block_size_x", "block_size_y", and "block_size_z".
            If more or different parameters divide the grid dimensions use
            grid_div_x/y/z options to specify this.

            You are allowed to use a string to specify the problem
            size. Within a string you are allowed to write Python
            arithmetic and use the names of tunable parameters as variables
            in these expressions.
            The Kernel Tuner will replace instances of the tunable parameters
            with their current value when computing the grid dimensions.
            See the reduction CUDA example for an example use of this feature.""",
            "string, int, or tuple(int or string, ..)")),
    ("arguments", ("""A list of kernel arguments, use numpy arrays for
            arrays, use numpy.int32 or numpy.float32 for scalars.""", "list")),
    ("grid_div_x", ("""A list of names of the parameters whose values divide
            the grid dimensions in the x-direction.
            The product of all grid divisor expressions is computed before dividing
            the problem_size in that dimension. Also note that the divison is treated
            as a float divison and resulting grid dimensions will be rounded up to
            the nearest integer number.

            Arithmetic expressions can be
            used if necessary inside the string containing a parameter name. For
            example, in some cases you may want to divide the problem size in the
            x-dimension with the number of warps rather than the number of threads
            in a block, in such cases one could use ["block_size_x/32"].

            If not supplied, ["block_size_x"] will be used by default, if you do not
            want any grid x-dimension divisors pass an empty list.""", "list")),
    ("grid_div_y", ("""A list of names of the parameters whose values divide
            the grid dimensions in the y-direction, ["block_size_y"] by default.
            If you do not want to divide the problem_size, you should pass an empty list.
            See grid_div_x for more details.""", "list")),
    ("grid_div_z", ("""A list of names of the parameters whose values divide
            the grid dimensions in the z-direction, ["block_size_z"] by default.
            If you do not want to divide the problem_size, you should pass an empty list.
            See grid_div_x for more details.""", "list")),
    ("cmem_args", ("""CUDA-specific feature for specifying constant memory
            arguments to the kernel. In OpenCL these are handled as normal
            kernel arguments, but in CUDA you can copy to a symbol. The way you
            specify constant memory arguments is by passing a dictionary with
            strings containing the constant memory symbol name together with numpy
            objects in the same way as normal kernel arguments.""",
            "dict(string: numpy object)")),
    ("block_size_names", ("""A list of strings that replace the defaults for the names
            that denote the thread block dimensions. If not passed, the behavior
            defaults to ``["block_size_x", "block_size_y", "block_size_z"]``""",
            "list(string)"))
    ])


_tuning_options = Options([
    ("tune_params", ("""A dictionary containing the parameter names as keys,
            and lists of possible parameter settings as values.
            The Kernel Tuner will try to compile and benchmark all possible
            combinations of all possible values for all tuning parameters.
            This typically results in a rather large search space of all
            possible kernel configurations.

            For each kernel configuration, each tuning parameter is
            replaced at compile-time with its current value.
            Currently, the Kernel Tuner uses the convention that the following
            list of tuning parameters are used as thread block dimensions:

                * "block_size_x"   thread block (work group) x-dimension
                * "block_size_y"   thread block (work group) y-dimension
                * "block_size_z"   thread block (work group) z-dimension

            Options for changing these defaults may be added later. If you
            don't want the thread block dimensions to be compiled in, you
            may use the built-in variables blockDim.xyz in CUDA or the
            built-in function get_local_size() in OpenCL instead.""",
            "dict( string : [...]")),
    ("restrictions", ("""A list of strings containing boolean expression that
        limit the search space in that they must be satisfied by the kernel
        configuration. These expressions must be true for the configuration
        to be part of the search space. For example:
        restrictions=["block_size_x==block_size_y*tile_size_y"] limits the
        search to configurations where the block_size_x equals the product
        of block_size_y and tile_size_y.
        The default is None.""", "list")),
    ("answer", ("""A list of arguments, similar to what you pass to arguments,
        that contains the expected output of the kernel after it has executed
        and contains None for each argument that is input-only. The expected
        output of the kernel will then be used to verify the correctness of
        each kernel in the parameter space before it will be benchmarked.""",
        "list")),
    ("atol", ("""The maximum allowed absolute difference between two elements
        in the output and the reference answer, as passed to numpy.allclose().
        Ignored if you have not passed a reference answer. Default value is
        1e-6, that is 0.000001.""", "float")),
    ("verify", ("""Python function used for output verification. By default,
        numpy.allclose is used for output verification, if this does not suit
        your application, you can pass a different function here.

        The function is expected to have two positional arguments. The first
        is the reference result, the second is the output computed by the
        kernel being verified. The types of these arguments depends on the
        type of the output arguments you are verifying. The function may also
        have an optional argument named atol, to which the value will be
        passed that was specified using the atol option to tune_kernel.
        The function should return True when the output passes the test, and
        False when the output fails the test.""", "func(ref, ans, atol=None)")),
    ("sample_fraction", ("""Benchmark only a sample fraction of the search space, False by
        default. To enable sampling, pass a value between 0 and 1. """, "float")),
    ("use_noodles", ("""Use Noodles workflow engine to tune in parallel using
        multiple threads, False by Default.
        Requires Noodles to be installed, use 'pip install noodles'.
        Note that Noodles requires Python 3.5 or newer.
        You can configure the number of threads to use with the option
        num_threads.""", "boolean")),
    ("num_threads", ("""The number of threads to use when using the Noodles
        workflow engine for tuning using multiple threads, 1 by default.
        Requires Noodles, see 'use_noodles' option.""", "int")),
    ("strategy", ("""Specify the strategy to use for searching through the
        parameter space, choose from:

            * "brute_force" (default),
            * "random_sample", specify: *sample_fraction*,
            * "minimize" or "basinhopping", specify: *method*,
            * "diff_evo", specify: *method*.

        "brute_force" is the default and iterates over the entire search
        space.

        "random_sample" can be used to only benchmark a fraction of the
        search space, specify a *sample_fraction* in the interval [0, 1].

        "minimize" and "basinhopping" strategies use minimizers to
        limit the search through the parameter space, select any of the
        methods: "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B",
        "TNC", "COBYLA", or "SLSQP". It is also possible to pass a
        function that implements a custom minimization strategy.

        "diff_evo" uses differential evolution and supports the following
        evolution strategies, which can be passed using the *method* argument:
        "best1bin", "best1exp", "rand1exp", "randtobest1exp", "best2exp",
        "rand2exp", "randtobest1bin", "best2bin", "rand2bin", "rand1bin".
        The default is "best1bin".

        """, "")),
    ("method", ("""Specify a method for the strategy that searches through
        the parameter space during tuning.

        When using strategy="minimize" or strategy="basinhopping", the
        following options are supported:
        "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B",
        "TNC", "COBYLA", or "SLSQP". It is also possible to pass a function
        that implements a custom minimization strategy.

        When using strategy="diff_evo", the following options are supported:
        "best1bin", "best1exp", "rand1exp", "randtobest1exp", "best2exp",
        "rand2exp", "randtobest1bin", "best2bin", "rand2bin", "rand1bin".

        """, "string or callable")),
    ("iterations", ("""The number of times a kernel should be executed and
        its execution time measured when benchmarking a kernel, 7 by default.""",
        "int")),
    ("verbose", ("""Sets whether or not to report about configurations that
        were skipped during the search. This could be due to several reasons:

            * kernel configuration fails one or more restrictions
            * too many threads per thread block
            * too much shared memory used by the kernel
            * too many resources requested for launch

        verbose is False by default.""", "bool")),
    ])

_device_options = Options([
    ("lang", ("""Specifies the language used for GPU kernels. The kernel_tuner
        automatically detects the language, but if it fails, you may specify
        the language using this argument, currently supported: "CUDA",
        "OpenCL", or "C".""", "string")),
    ("device", ("""CUDA/OpenCL device to use, in case you have multiple
        CUDA-capable GPUs or OpenCL devices you may use this to select one,
        0 by default. Ignored if you are tuning host code by passing
        lang="C".""", "int")),
    ("platform", ("""OpenCL platform to use, in case you have multiple
        OpenCL platforms you may use this to select one,
        0 by default. Ignored if not using OpenCL. """, "int")),
    ("quiet", ("""Control whether or not to print to the console which
        device is being used, False by default""", "boolean")),
    ("compiler", ("""A string containing your preferred compiler,
        only effective with lang="C". """, "string")),
    ("compiler_options", ("""A list of strings that specify compiler
        options.""", "list(string)"))
    ])



def _get_docstring(opts):
    docstr = ""
    for k, v in opts.items():
        docstr += "    :param " + k + ": " + v[0] + "\n"
        docstr += "    :type "  + k + ": " + v[1] + "\n\n"
    return docstr

_tune_kernel_docstring = """ Tune a CUDA kernel given a set of tunable parameters

%s

    :returns: A list of dictionaries of all executed kernel configurations and their
        execution times. And a dictionary with information about the environment
        in which the tuning took place. This records device name, properties,
        version info, and so on.
    :rtype: list(dict()), dict()

""" % _get_docstring(_kernel_options) + _get_docstring(_tuning_options) + _get_docstring(_device_options)

#"""

def tune_kernel(kernel_name, kernel_string, problem_size, arguments,
                tune_params, grid_div_x=None, grid_div_y=None, grid_div_z=None,
                restrictions=None, answer=None, atol=1e-6, verify=None, verbose=False,
                lang=None, device=0, platform=0, cmem_args=None,
                num_threads=1, use_noodles=False, sample_fraction=False, compiler=None, compiler_options=None, log=None,
                iterations=7, block_size_names=None, quiet=False, strategy=None, method=None):

    if log:
        logging.basicConfig(filename=kernel_name + datetime.now().strftime('%Y%m%d-%H:%M:%S') + '.log', level=log)

    #see if the kernel arguments have correct type
    util.check_argument_list(arguments)

    #sort all the options into separate dicts
    opts = locals()
    kernel_options = Options([(k, opts[k]) for k in _kernel_options.keys()])
    tuning_options = Options([(k, opts[k]) for k in _tuning_options.keys()])
    device_options = Options([(k, opts[k]) for k in _device_options.keys()])

    logging.debug('tune_kernel called')
    logging.debug('kernel_options: %s', util.get_config_string(kernel_options))
    logging.debug('tuning_options: %s', util.get_config_string(tuning_options))
    logging.debug('device_options: %s', util.get_config_string(device_options))

    #select strategy based on user options
    if sample_fraction and not strategy in [None, 'sample_fraction']:
        raise ValueError("It's not possible to use both sample_fraction in combination with other strategies. " \
                         'Please set strategy=None or strategy="random_sample", when using sample_fraction')

    if strategy in [None, 'sample_fraction', 'brute_force']:
        if sample_fraction:
            use_strategy = random_sample
        else:
            use_strategy = brute_force
    elif strategy in ["minimize", "basinhopping"]:
        if method:
            if not (method in ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B",
                               "TNC", "COBYLA", "SLSQP"] or callable(method)):
                raise ValueError("method option not recognized")
        else:
            method = "L-BFGS-B"
        if strategy == "minimize":
            use_strategy = minimize
        else:
            use_strategy = basinhopping
    elif strategy == "diff_evo":
        use_strategy = diff_evo
        if method:
            if not method in ["best1bin", "best1exp", "rand1exp", "randtobest1exp", "best2exp",
                              "rand2exp", "randtobest1bin", "best2bin", "rand2bin", "rand1bin"]:
                raise ValueError("method option not recognized")
    else:
        raise ValueError("strategy option not recognized")
    strategy = use_strategy

    #select runner based on user options
    if num_threads == 1 and not use_noodles:
        from kernel_tuner.runners.sequential import SequentialRunner
        runner = SequentialRunner(kernel_options, device_options, iterations)
    elif num_threads > 1 and not use_noodles:
        raise ValueError("Using multiple threads requires the Noodles runner, use use_noodles=True")
    elif use_noodles:
        #check if Python version matches required by Noodles
        if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 5):
            raise ValueError("Using multiple threads requires Noodles, Noodles requires Python 3.5 or higher")
        #check if noodles is installed in a way that works with Python 3.4 or newer
        noodles_installed = importlib.util.find_spec("noodles") is not None
        if not noodles_installed:
            raise ValueError("Using multiple threads requires Noodles, please use 'pip install noodles'")
        #import the NoodlesRunner
        from kernel_tuner.runners.noodles import NoodlesRunner
        runner = NoodlesRunner(device_options, num_threads)
    else:
        raise ValueError("Somehow no runner was selected, this should not happen, please file a bug report")

    #call the strategy to execute the tuning process
    results, env = strategy.tune(runner, kernel_options, device_options, tuning_options)

    #finished iterating over search space
    if results:     #checks if results is not empty
        best_config = min(results, key=lambda x: x['time'])
        print("best performing configuration:", util.get_config_string(best_config))
    else:
        print("no results to report")

    del runner.dev

    return results, env


tune_kernel.__doc__ = _tune_kernel_docstring


_run_kernel_docstring = """Compile and run a single kernel

    Compiles and runs a single kernel once, given a specific instance of the kernels tuning parameters.
    However, instead of measuring execution time run_kernel returns the output of the kernel.
    The output is returned as a list of numpy arrays that contains the state of all the kernel arguments
    after execution on the GPU.

    To summarize what this function will do for you in one call:
     * Compile the kernel according to the set of parameters passed
     * Allocate GPU memory to hold all kernel arguments
     * Move the all data to the GPU
     * Execute the kernel on the GPU
     * Copy all data from the GPU back to the host and return it as a list of Numpy arrays

    This function was added to the Kernel Tuner mostly to allow easy testing for kernel correctness.
    On purpose, the interface is a lot like `tune_kernel()`.

%s

    :param params: A dictionary containing the tuning parameter names as keys
            and a single value per tuning parameter as values.
    :type params: dict( string: int )

    :returns: A list of numpy arrays, similar to the arguments passed to this
        function, containing the output after kernel execution.
    :rtype: list
""" % _get_docstring(_kernel_options) + _get_docstring(_device_options)


def run_kernel(kernel_name, kernel_string, problem_size, arguments,
               params, grid_div_x=None, grid_div_y=None, grid_div_z=None,
               lang=None, device=0, platform=0, cmem_args=None, compiler=None, compiler_options=None,
               block_size_names=None, quiet=False):

    #sort options into separate dicts
    opts = locals()
    kernel_options = Options([(k, opts[k]) for k in _kernel_options.keys()])
    device_options = Options([(k, opts[k]) for k in _device_options.keys()])

    #detect language and create the right device function interface
    dev = core.DeviceInterface(kernel_string, iterations=1, **device_options)

    #move data to the GPU
    util.check_argument_list(arguments)
    gpu_args = dev.ready_argument_list(arguments)

    instance = None
    try:
        #create kernel instance
        instance = dev.create_kernel_instance(kernel_options, params, False)
        if instance is None:
            raise Exception("cannot create kernel instance, too many threads per block")

        #compile the kernel
        func = dev.compile_kernel(instance, False)
        if func is None:
            raise Exception("cannot compile kernel, too much shared memory used")

        #add constant memory arguments to compiled module
        if cmem_args is not None:
            dev.copy_constant_memory_args(cmem_args)
    finally:
        #delete temp files
        if instance is not None:
            for v in instance.temp_files.values():
                util.delete_temp_file(v)

    #run the kernel
    if not dev.run_kernel(func, gpu_args, instance):
        raise Exception("runtime error occured, too many resources requested")

    #copy data in GPU memory back to the host
    results = []
    for i, arg in enumerate(arguments):
        if numpy.isscalar(arg):
            results.append(arg)
        else:
            results.append(numpy.zeros_like(arg))
            dev.memcpy_dtoh(results[-1], gpu_args[i])

    return results


run_kernel.__doc__ = _run_kernel_docstring
