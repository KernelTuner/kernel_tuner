"""Kernel Tuner interface module

This module contains the main functions that Kernel Tuner
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

import json
import os.path
from collections import OrderedDict
import importlib
from datetime import datetime
import logging
import sys
import numpy

import kernel_tuner.util as util
import kernel_tuner.core as core

from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner.runners.simulation import SimulationRunner

try:
    import torch
except ImportError:
    torch = util.TorchPlaceHolder()

from kernel_tuner.strategies import brute_force, random_sample, diff_evo, minimize, basinhopping, genetic_algorithm, mls, pso, simulated_annealing, firefly_algorithm, bayes_opt

strategy_map = {
    "brute_force": brute_force,
    "random_sample": random_sample,
    "minimize": minimize,
    "basinhopping": basinhopping,
    "diff_evo": diff_evo,
    "genetic_algorithm": genetic_algorithm,
    "mls": mls,
    "pso": pso,
    "simulated_annealing": simulated_annealing,
    "firefly_algorithm": firefly_algorithm,
    "bayes_opt": bayes_opt,
}


class Options(OrderedDict):
    """read-only class for passing options around"""

    def __getattr__(self, name):
        if not name.startswith('_'):
            return self[name]
        return super(Options, self).__getattr__(name)

    def __deepcopy__(self, _):
        return self


_kernel_options = Options([("kernel_name", ("""The name of the kernel in the code.""", "string")),
                           ("kernel_source", ("""The CUDA, OpenCL, or C kernel code.
            It is allowed for the code to be passed as a string, a filename, a function
            that returns a string of code, or a list when the code needs auxilliary files.

            To support combined host and device code tuning, a list of
            filenames can be passed. The first file in the list should be the
            file that contains the host code. The host code is assumed to
            include or read in any of the files in the list beyond the first.
            The tunable parameters can be used within all files.

            Another alternative is to pass a code generating function.
            The purpose of this is to support the use of code generating
            functions that generate the kernel code based on the specific
            parameters. This function should take one positional argument,
            which will be used to pass a dict containing the parameters.
            The function should return a string with the source code for
            the kernel.""", "string or list and/or callable")),
                           ("lang", ("""Specifies the language used for GPU kernels. The kernel_tuner
        automatically detects the language, but if it fails, you may specify
        the language using this argument, currently supported: "CUDA", "Cupy",
        "OpenCL", or "C".""", "string")),
                           ("problem_size", ("""The size of the domain from which the grid dimensions
            of the kernel are computed.

            This can be specified using an int, string, function, or
            1,2,3-dimensional tuple.

            In general, do not divide the problem_size yourself by the thread block sizes.
            Kernel Tuner does this for you based on tunable parameters,
            called "block_size_x", "block_size_y", and "block_size_z".
            If more or different parameters divide the grid dimensions use
            grid_div_x/y/z options to specify this.

            In most use-cases the problem_size is specified using a single integer
            or a tuple of integers,
            but Kernel Tuner supports more advanced use cases where the problem_size
            itself depends on the tunable parameters in some way.

            You are allowed to use a function or string to specify the problem_size.
            A function should accept a dictionary with the tunable parameters
            for this kernel configuration and directly return a tuple
            that specifies the problem size in all dimensions.

            When passing a string, you are allowed to write Python
            arithmetic and use the names of tunable parameters as variables
            in these expressions. Kernel Tuner will replace instances of the tunable
            parameters with their current value when computing the grid dimensions.
            This option exists for convenience, but do note that using a lambda
            function is probably safer. The string notation should only return
            the problem size for one dimension, but can be used inside
            a tuple, possibly in combination with integers or more strings in
            different dimensions.

            See the reduction CUDA example for an example use of this feature.""", "callable, string, int, or tuple(int or string, ..)")),
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
            in a block, in such cases one could for example use ["block_size_x/32"].
            Another option is to pass a function to grid_div_x that accepts a
            dictionary with the tunable parameters and returns the grid divisor
            in this dimension, for example: grid_div_x=lambda p:p["block_size_x"]/32.

            If not supplied, ["block_size_x"] will be used by default, if you do not
            want any grid x-dimension divisors pass an empty list.""", "callable or list")),
                           ("grid_div_y", ("""A list of names of the parameters whose values divide
            the grid dimensions in the y-direction, ["block_size_y"] by default.
            If you do not want to divide the problem_size, you should pass an empty list.
            See grid_div_x for more details.""", "list")),
                           ("grid_div_z", ("""A list of names of the parameters whose values divide
            the grid dimensions in the z-direction, ["block_size_z"] by default.
            If you do not want to divide the problem_size, you should pass an empty list.
            See grid_div_x for more details.""", "list")),
                           ("smem_args", ("""CUDA-specific feature for specifying shared memory options
            to the kernel. At the moment only 'size' is supported, but setting the
            shared memory configuration on Kepler GPUs for example could be added
            in the future. Size should denote the number of bytes for to use when
            dynamically allocating shared memory.""", "dict(string: numpy object)")),
                           ("cmem_args", ("""CUDA-specific feature for specifying constant memory
            arguments to the kernel. In OpenCL these are handled as normal
            kernel arguments, but in CUDA you can copy to a symbol. The way you
            specify constant memory arguments is by passing a dictionary with
            strings containing the constant memory symbol name together with numpy
            objects in the same way as normal kernel arguments.""", "dict(string: numpy object)")),
                           ("texmem_args", ("""CUDA-specific feature for specifying texture memory
            arguments to the kernel. You specify texture memory arguments by passing a
            dictionary with strings containing the texture reference name together with
            the texture contents. These contents can be either simply a numpy object,
            or a dictionary containing the numpy object under the key 'array' plus the
            configuration options 'filter_mode' ('point' or 'linear), 'address_mode'
            (a list of 'border', 'clamp', 'mirror', 'wrap' per axis),
            'normalized_coordinates' (True/False).""", "dict(string: numpy object or dict)")),
                           ("block_size_names", ("""A list of strings that replace the defaults for the names
            that denote the thread block dimensions. If not passed, the behavior
            defaults to ``["block_size_x", "block_size_y", "block_size_z"]``""", "list(string)"))])

_tuning_options = Options([("tune_params", ("""A dictionary containing the parameter names as keys,
            and lists of possible parameter settings as values.
            Kernel Tuner will try to compile and benchmark all possible
            combinations of all possible values for all tuning parameters.
            This typically results in a rather large search space of all
            possible kernel configurations.

            For each kernel configuration, each tuning parameter is
            replaced at compile-time with its current value.
            Currently, Kernel Tuner uses the convention that the following
            list of tuning parameters are used as thread block dimensions:

                * "block_size_x"   thread block (work group) x-dimension
                * "block_size_y"   thread block (work group) y-dimension
                * "block_size_z"   thread block (work group) z-dimension

            Options for changing these defaults may be added later. If you
            don't want the thread block dimensions to be compiled in, you
            may use the built-in variables blockDim.xyz in CUDA or the
            built-in function get_local_size() in OpenCL instead.""", "dict( string : [...]")),
                           ("restrictions", ("""An option to limit the search space with restrictions.
        The restrictions can be specified using a function or a list of strings.
        The function should take one argument, namely a dictionary with the
        tunable parameters of the kernel configuration, if the function returns
        True the configuration is considered to be part of the search space, or
        False otherwise.
        The other way to specify restrictions is using a list of strings
        containing boolean expression that must be satisfied by the kernel
        configuration. These expressions must all be true for the configuration
        to be part of the search space. For example:
        restrictions=["block_size_x==block_size_y*tile_size_y"] limits the
        search to configurations where the block_size_x equals the product
        of block_size_y and tile_size_y.
        The default is None.""", "callable or list(strings)")),
                           ("answer", ("""A list of arguments, similar to what you pass to arguments,
        that contains the expected output of the kernel after it has executed
        and contains None for each argument that is input-only. The expected
        output of the kernel will then be used to verify the correctness of
        each kernel in the parameter space before it will be benchmarked.""", "list")),
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
                           ("strategy", ("""Specify the strategy to use for searching through the
        parameter space, choose from:

            * "brute_force" (default),
            * "random_sample", specify: *sample_fraction*,
            * "minimize" or "basinhopping", specify: *method*,
            * "diff_evo", specify: *method*.
            * "genetic_algorithm"
            * "pso"
            * "firefly_algorithm"
            * "simulated_annealing"
            * "bayes_opt"

        "brute_force" is the default and iterates over the entire search
        space.

        "random_sample" can be used to only benchmark a fraction of the
        search space, specify a *sample_fraction* in the interval [0, 1].

        "minimize" and "basinhopping" strategies use minimizers to
        limit the search through the parameter space.

        "diff_evo" uses differential evolution.

        "genetic_algorithm" implements a Genetic Algorithm, default
        setting uses a population size of 20 for 100 generations.

        "pso" implements Particle Swarm Optimization, using the default
        setting of 20 particles for 100 iterations.

        "firefly_algorithm" implements the Firefly Algorithm, using 20
        fireflies for 100 iterations.

        "simulated_annealing" uses Simulated Annealing.

        "bayes_opt" uses Bayesian Optimization.

        """, "")),
                           ("strategy_options", ("""A dict with options for the tuning strategy

        Example usage:

         * strategy="basinhopping",
           strategy_options={"method": "BFGS",
                             "maxiter": 100,
                             "T": 1.0}
         * strategy="diff_evo",
           strategy_options={"method": "best1bin",
                             "popsize": 20}
         * strategy="genetic_algorithm",
           strategy_options={"method": "uniform",
                             "popsize": 20,
                             "maxiter": 100}

        strategy="minimize" and strategy="basinhopping", support the following
        options for "method":
        "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B",
        "TNC", "COBYLA", or "SLSQP". It is also possible to pass a function
        that implements a custom minimization strategy.
        The default is "L-BFGS-B".
        strategy="basinhopping" also supports "T", which is 1.0 by default.

        strategy="diff_evo" supports the following creation "method" options:
        "best1bin", "best1exp", "rand1exp", "randtobest1exp", "best2exp",
        "rand2exp", "randtobest1bin", "best2bin", "rand2bin", "rand1bin".
        The default is "best1bin".

        strategy="genetic_algorithm" uses "method" to select the crossover
        method, options are: "single_point", "two_point", "uniform", and
        "disruptive_uniform".
        The default is "uniform".
        Also "mutation_chance" can be set to control the chance of a mutation,
        which is separately evaluated for each dimension. For example, set
        to 100 for a probability of 0.01 of a mutation per tunable parameter.

        strategy="random_sample" supports "fraction" to specify
        the fraction of the search space to sample in the interval [0,1].

        strategy="firefly_algorithm" supports the following parameters:
        B0 = 1.0, gamma = 1.0, alpha = 0.20.

        strategy="simulated_annealing" supports parameters:
        T = 1.0, T_min = 0.001, alpha = 0.9.

        strategy="bayes_opt" supports acquisition methods: "poi" (default),
        "ei", "ucb". And parameters, popsize (initial random guesses),
        maxiter, alpha, kappa, xi.

        "maxiter" is supported by "minimize", "basinhopping", "diff_evo"
        "firefly_algorithm", "pso", "genetic_algorithm", "bayes_opt". Note
        that maxiter generally refers to iterations of the strategy, not
        the maximum number of function evaluations.

    """, "dict")),
                           ("iterations", ("""The number of times a kernel should be executed and
        its execution time measured when benchmarking a kernel, 7 by default.""", "int")),
                           ("verbose", ("""Sets whether or not to report about configurations that
        were skipped during the search. This could be due to several reasons:

            * kernel configuration fails one or more restrictions
            * too many threads per thread block
            * too much shared memory used by the kernel
            * too many resources requested for launch

        verbose is False by default.""", "bool")),
                           ("cache", ("""filename for caching/logging benchmarked instances
        filename uses suffix ".json"
        if the file exists it is read and tuning continues from this file
        """, "string")), ("metrics", ("specifies user-defined metrics", "OrderedDict")),
                           ("simulation_mode", ("Simulate an auto-tuning search from an existing cachefile", "bool")),
                           ("observers", ("""A list of BenchmarkObservers""", "list"))])

_device_options = Options([("device", ("""CUDA/OpenCL device to use, in case you have multiple
        CUDA-capable GPUs or OpenCL devices you may use this to select one,
        0 by default. Ignored if you are tuning host code by passing
        lang="C".""", "int")),
                           ("platform", ("""OpenCL platform to use, in case you have multiple
        OpenCL platforms you may use this to select one,
        0 by default. Ignored if not using OpenCL. """, "int")),
                           ("quiet", ("""Control whether or not to print to the console which
        device is being used, False by default""", "boolean")),
                           ("compiler", ("""A string containing your preferred compiler,
        only effective with lang="C". """, "string")), ("compiler_options", ("""A list of strings that specify compiler
        options.""", "list(string)"))])


def _get_docstring(opts):
    docstr = ""
    for k, v in opts.items():
        docstr += "    :param " + k + ": " + v[0] + "\n"
        docstr += "    :type " + k + ": " + v[1] + "\n\n"
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


def tune_kernel(kernel_name, kernel_string, problem_size, arguments, tune_params, grid_div_x=None, grid_div_y=None, grid_div_z=None, restrictions=None,
                answer=None, atol=1e-6, verify=None, verbose=False, lang=None, device=0, platform=0, smem_args=None, cmem_args=None, texmem_args=None,
                compiler=None, compiler_options=None, log=None, iterations=7, block_size_names=None, quiet=False, strategy=None, strategy_options=None,
                cache=None, metrics=None, simulation_mode=False, observers=None):

    if log:
        logging.basicConfig(filename=kernel_name + datetime.now().strftime('%Y%m%d-%H:%M:%S') + '.log', level=log)

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang)

    _check_user_input(kernel_name, kernel_source, arguments, block_size_names)

    # check for forbidden names in tune parameters
    util.check_tune_params_list(tune_params)

    # check whether block_size_names are used as expected
    util.check_block_size_params_names_list(block_size_names, tune_params)

    if iterations < 1:
        raise ValueError("Iterations should be at least one!")

    #sort all the options into separate dicts
    opts = locals()
    kernel_options = Options([(k, opts[k]) for k in _kernel_options.keys()])
    tuning_options = Options([(k, opts[k]) for k in _tuning_options.keys()])
    device_options = Options([(k, opts[k]) for k in _device_options.keys()])
    tuning_options["snap"] = True

    logging.debug('tune_kernel called')
    logging.debug('kernel_options: %s', util.get_config_string(kernel_options))
    logging.debug('tuning_options: %s', util.get_config_string(tuning_options))
    logging.debug('device_options: %s', util.get_config_string(device_options))

    if strategy:
        if strategy in strategy_map:
            strategy = strategy_map[strategy]
        else:
            raise ValueError("Strategy %s not recognized" % strategy)

        #make strategy_options into an Options object
        if tuning_options.strategy_options:
            if not isinstance(strategy_options, Options):
                tuning_options.strategy_options = Options(strategy_options)

            #select strategy based on user options
            if "fraction" in tuning_options.strategy_options and not tuning_options.strategy == 'random_sample':
                raise ValueError('It is not possible to use fraction in combination with strategies other than "random_sample". ' \
                                 'Please set strategy="random_sample", when using "fraction" in strategy_options')

            #check if method is supported by the selected strategy
            if "method" in tuning_options.strategy_options:
                method = tuning_options.strategy_options.method
                if not method in strategy.supported_methods:
                    raise ValueError('Method %s is not supported for strategy %s' % (method, tuning_options.strategy))

        #if no strategy_options dict has been passed, create empty dictionary
        else:
            tuning_options.strategy_options = Options({})

    #if no strategy selected
    else:
        strategy = brute_force

    # select the runner for this job based on input
    selected_runner = SimulationRunner if simulation_mode is True else SequentialRunner
    with selected_runner(kernel_source, kernel_options, device_options, iterations, observers) as runner:

        #the user-specified function may or may not have an optional atol argument;
        #we normalize it so that it always accepts atol.
        tuning_options.verify = util.normalize_verify_function(tuning_options.verify)

        #process cache
        if cache:
            if cache[-5:] != ".json":
                cache += ".json"

            util.process_cache(cache, kernel_options, tuning_options, runner)
        else:
            tuning_options.cache = {}
            tuning_options.cachefile = None

        #call the strategy to execute the tuning process
        results, env = strategy.tune(runner, kernel_options, device_options, tuning_options)

        #finished iterating over search space
        if not device_options.quiet:
            if results:    #checks if results is not empty
                best_config = min(results, key=lambda x: x['time'])
                units = getattr(runner, "units", None)
                print("best performing configuration:")
                util.print_config_output(tune_params, best_config, device_options.quiet, metrics, units)
            else:
                print("no results to report")

        if cache:
            util.close_cache(cache)

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

    This function was added to Kernel Tuner mostly to allow easy testing for kernel correctness.
    On purpose, the interface is a lot like `tune_kernel()`.

%s

    :param params: A dictionary containing the tuning parameter names as keys
            and a single value per tuning parameter as values.
    :type params: dict( string: int )

    :returns: A list of numpy arrays, similar to the arguments passed to this
        function, containing the output after kernel execution.
    :rtype: list
""" % _get_docstring(_kernel_options) + _get_docstring(_device_options)


def run_kernel(kernel_name, kernel_string, problem_size, arguments, params, grid_div_x=None, grid_div_y=None, grid_div_z=None, lang=None, device=0, platform=0,
               smem_args=None, cmem_args=None, texmem_args=None, compiler=None, compiler_options=None, block_size_names=None, quiet=False, log=None):

    if log:
        logging.basicConfig(filename=kernel_name + datetime.now().strftime('%Y%m%d-%H:%M:%S') + '.log', level=log)

    kernel_source = core.KernelSource(kernel_name, kernel_string, lang)

    _check_user_input(kernel_name, kernel_source, arguments, block_size_names)

    #sort options into separate dicts
    opts = locals()
    kernel_options = Options([(k, opts[k]) for k in _kernel_options.keys()])
    device_options = Options([(k, opts[k]) for k in _device_options.keys()])

    #detect language and create the right device function interface
    with core.DeviceInterface(kernel_source, iterations=1, **device_options) as dev:

        #move data to the GPU
        gpu_args = dev.ready_argument_list(arguments)

        instance = None
        try:
            #create kernel instance
            instance = dev.create_kernel_instance(kernel_source, kernel_options, params, False)
            if instance is None:
                raise Exception("cannot create kernel instance, too many threads per block")

            # see if the kernel arguments have correct type
            util.check_argument_list(instance.name, instance.kernel_string, arguments)

            #compile the kernel
            func = dev.compile_kernel(instance, False)
            if func is None:
                raise Exception("cannot compile kernel, too much shared memory used")

            #add shared memory arguments to compiled module
            if smem_args is not None:
                dev.copy_shared_memory_args(util.get_smem_args(smem_args, params))
            #add constant memory arguments to compiled module
            if cmem_args is not None:
                dev.copy_constant_memory_args(cmem_args)
            #add texture memory arguments to compiled module
            if texmem_args is not None:
                dev.copy_texture_memory_args(texmem_args)
        finally:
            #delete temp files
            if instance is not None:
                instance.delete_temp_files()

        #run the kernel
        if not dev.run_kernel(func, gpu_args, instance):
            raise Exception("runtime error occured, too many resources requested")

        #copy data in GPU memory back to the host
        results = []
        for i, arg in enumerate(arguments):
            if numpy.isscalar(arg):
                results.append(arg)
            elif isinstance(arg, torch.Tensor):
                results.append(arg.cpu())
            else:
                results.append(numpy.zeros_like(arg))
                dev.memcpy_dtoh(results[-1], gpu_args[i])

    return results


run_kernel.__doc__ = _run_kernel_docstring


def _check_user_input(kernel_name, kernel_source, arguments, block_size_names):
    # see if the kernel arguments have correct type
    kernel_source.check_argument_lists(kernel_name, arguments)

    # check for types and length of block_size_names
    util.check_block_size_names(block_size_names)
