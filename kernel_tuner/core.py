""" Module for grouping the core functionality needed by most runners """

import logging
import re
import time
from collections import namedtuple

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np

import kernel_tuner.util as util
from kernel_tuner.accuracy import Tunable
from kernel_tuner.backends.pycuda import PyCudaFunctions
from kernel_tuner.backends.cupy import CupyFunctions
from kernel_tuner.backends.hip import HipFunctions
from kernel_tuner.backends.nvcuda import CudaFunctions
from kernel_tuner.backends.opencl import OpenCLFunctions
from kernel_tuner.backends.compiler import CompilerFunctions
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.tegra import TegraObserver
from kernel_tuner.observers.observer import ContinuousObserver, OutputObserver, PrologueObserver

try:
    import torch
except ImportError:
    torch = util.TorchPlaceHolder()

try:
    from hip._util.types import DeviceArray
except ImportError:
    DeviceArray = Exception # using Exception here as a type that will never be among kernel arguments

_KernelInstance = namedtuple(
    "_KernelInstance",
    [
        "name",
        "kernel_source",
        "kernel_string",
        "temp_files",
        "threads",
        "grid",
        "params",
        "arguments",
    ],
)


class KernelInstance(_KernelInstance):
    """Class that represents the specific parameterized instance of a kernel"""

    def delete_temp_files(self):
        """Delete any generated temp files"""
        for v in self.temp_files.values():
            util.delete_temp_file(v)

    def prepare_temp_files_for_error_msg(self):
        """Prepare temp file with source code, and return list of temp file names"""
        temp_filename = util.get_temp_filename(suffix=self.kernel_source.get_suffix())
        util.write_file(temp_filename, self.kernel_string)
        ret = [temp_filename]
        ret.extend(self.temp_files.values())
        return ret


class KernelSource(object):
    """Class that holds the kernel sources.

    There is a primary kernel source, which can be either a source string,
    a filename (indicating a file containing the kernel source code),
    or a callable (generating the kernel source code).
    There can additionally be (one or multiple) secondary kernel sources, which
    must be filenames.
    """

    def __init__(self, kernel_name, kernel_sources, lang, defines=None):
        if not isinstance(kernel_sources, list):
            kernel_sources = [kernel_sources]
        self.kernel_sources = kernel_sources
        self.kernel_name = kernel_name
        self.defines = defines
        if lang is None:
            if callable(self.kernel_sources[0]):
                raise TypeError(
                    "Please specify language when using a code generator function"
                )
            kernel_string = self.get_kernel_string(0)
            lang = util.detect_language(kernel_string)

        # The validity of lang is checked later, when creating the DeviceInterface
        self.lang = lang.upper()

    def get_kernel_string(self, index=0, params=None):
        """retrieve the kernel source with the given index and return as a string

        See util.get_kernel_string() for details.

        :param index: Index of the kernel source in the list of sources.
        :type index: int

        :param params: Dictionary containing the tunable parameters for this specific
            kernel instance, only needed when kernel_source is a generator.
        :type param: dict

        :returns: A string containing the kernel code.
        :rtype: string
        """
        logging.debug("get_kernel_string called")

        kernel_source = self.kernel_sources[index]
        return util.get_kernel_string(kernel_source, params)

    def prepare_list_of_files(
        self, kernel_name, params, grid, threads, block_size_names
    ):
        """prepare the kernel string along with any additional files

        The first file in the list is allowed to include or read in the others
        The files beyond the first are considered additional files that may also contain tunable parameters

        For each file beyond the first this function creates a temporary file with
        preprocessors statements inserted. Occurrences of the original filenames in the
        first file are replaced with their temporary counterparts.

        :param kernel_name: A string specifying the kernel name.
        :type kernel_name: string

        :param params: A dictionary with the tunable parameters for this particular
            instance.
        :type params: dict()

        :param grid: The grid dimensions for this instance. The grid dimensions are
            also inserted into the code as if they are tunable parameters for
            convenience.
        :type grid: tuple()

        :param threads: The thread block dimensions for this instance. The thread block are
            also inserted into the code as if they are tunable parameters for
            convenience.
        :type threads: tuple()

        :param block_size_names: A list of strings that denote the names
            for the thread block dimensions.
        :type block_size_names: list(string)

        """
        temp_files = dict()

        for i, f in enumerate(self.kernel_sources):
            if i > 0 and not util.looks_like_a_filename(f):
                raise ValueError(
                    "When passing multiple kernel sources, the secondary entries must be filenames"
                )

            ks = self.get_kernel_string(i, params)
            # add preprocessor statements
            n, ks = util.prepare_kernel_string(
                kernel_name,
                ks,
                params,
                grid,
                threads,
                block_size_names,
                self.lang,
                self.defines,
            )

            if i == 0:
                # primary kernel source
                name = n
                kernel_string = ks
                continue

            # save secondary kernel sources to temporary files

            # generate temp filename with the same extension
            temp_file = util.get_temp_filename(suffix="." + f.split(".")[-1])
            temp_files[f] = temp_file
            util.write_file(temp_file, ks)
            # replace occurrences of the additional file's name in the first kernel_string with the name of the temp file
            kernel_string = kernel_string.replace(f, temp_file)

        return name, kernel_string, temp_files

    def get_user_suffix(self, index=0):
        """Get the suffix of the kernel filename, if the user specified one. Return None otherwise."""
        if util.looks_like_a_filename(self.kernel_sources[index]) and (
            "." in self.kernel_sources[index]
        ):
            return "." + self.kernel_sources[index].split(".")[-1]
        return None

    def get_suffix(self, index=0):
        """Return a suitable suffix for a kernel filename.

        This uses the user-specified suffix if available, or one based on the
        lang/backend otherwise.
        """

        # TODO: Consider delegating this to the backend
        suffix = self.get_user_suffix(index)
        if suffix is not None:
            return suffix

        _suffixes = {"CUDA": ".cu", "OpenCL": ".cl", "C": ".c"}
        try:
            return _suffixes[self.lang]
        except KeyError:
            return ".c"

    def check_argument_lists(self, kernel_name, arguments):
        """Check if the kernel arguments have the correct types

        This is done by calling util.check_argument_list on each kernel string.
        """
        for i, f in enumerate(self.kernel_sources):
            if not callable(f):
                util.check_argument_list(
                    kernel_name, self.get_kernel_string(i), arguments
                )
            else:
                logging.debug(
                    "Checking of arguments list not supported yet for code generators."
                )


class DeviceInterface(object):
    """Class that offers a High-Level Device Interface to the rest of the Kernel Tuner"""

    def __init__(
        self,
        kernel_source,
        device=0,
        platform=0,
        quiet=False,
        compiler=None,
        compiler_options=None,
        iterations=7,
        observers=None,
    ):
        """Instantiate the DeviceInterface, based on language in kernel source

        :param kernel_source: The kernel sources
        :type kernel_source: kernel_tuner.core.KernelSource

        :param device: CUDA/OpenCL device to use, in case you have multiple
            CUDA-capable GPUs or OpenCL devices you may use this to select one,
            0 by default. Ignored if you are tuning host code by passing lang="C".
        :type device: int

        :param platform: OpenCL platform to use, in case you have multiple
            OpenCL platforms you may use this to select one,
            0 by default. Ignored if not using OpenCL.
        :type device: int

        :param lang: Specifies the language used for GPU kernels.
            Currently supported: "CUDA", "OpenCL", "HIP" or "C"
        :type lang: string

        :param compiler_options: The compiler options to use when compiling kernels for this device.
        :type compiler_options: list of strings

        :param iterations: Number of iterations to be used when benchmarking using this device.
        :type iterations: int

        :param times: Return the execution time of all iterations.
        :type times: bool

        """
        lang = kernel_source.lang

        logging.debug("DeviceInterface instantiated, lang=%s", lang)

        if lang.upper() == "CUDA":
            dev = PyCudaFunctions(
                device,
                compiler_options=compiler_options,
                iterations=iterations,
                observers=observers,
            )
        elif lang.upper() == "CUPY":
            dev = CupyFunctions(
                device,
                compiler_options=compiler_options,
                iterations=iterations,
                observers=observers,
            )
        elif lang.upper() == "NVCUDA":
            dev = CudaFunctions(
                device,
                compiler_options=compiler_options,
                iterations=iterations,
                observers=observers,
            )
        elif lang.upper() == "OPENCL":
            dev = OpenCLFunctions(
                device,
                platform,
                compiler_options=compiler_options,
                iterations=iterations,
                observers=observers,
            )
        elif lang.upper() in ["C", "FORTRAN"]:
            dev = CompilerFunctions(
                compiler=compiler,
                compiler_options=compiler_options,
                iterations=iterations,
            )
        elif lang.upper() == "HIP":
            dev = HipFunctions(
                device,
                compiler_options=compiler_options,
                iterations=iterations,
                observers=observers,
            )
        else:
            raise ValueError("Sorry, support for languages other than CUDA, OpenCL, HIP, C, and Fortran is not implemented yet")
        self.dev = dev

        # look for NVMLObserver and TegraObserver in observers, if present, enable special tunable parameters through nvml/tegra
        self.use_nvml = False
        self.use_tegra = False
        self.continuous_observers = []
        self.output_observers = []
        self.prologue_observers = []
        if observers:
            for obs in observers:
                if isinstance(obs, NVMLObserver):
                    self.nvml = obs.nvml
                    self.use_nvml = True
                if isinstance(obs, TegraObserver):
                    self.tegra = obs.tegra
                    self.use_tegra = True
                if hasattr(obs, "continuous_observer"):
                    self.continuous_observers.append(obs.continuous_observer)
                if isinstance(obs, OutputObserver):
                    self.output_observers.append(obs)
                if isinstance(obs, PrologueObserver):
                    self.prologue_observers.append(obs)

        # Take list of observers from self.dev because Backends tend to add their own observer
        self.benchmark_observers = [
            obs for obs in self.dev.observers if not isinstance(obs, (ContinuousObserver, PrologueObserver))
        ]

        self.iterations = iterations

        self.lang = lang
        self.units = dev.units
        self.name = dev.name
        self.max_threads = dev.max_threads
        if not quiet:
            print("Using: " + self.dev.name)

    def benchmark_prologue(self, func, gpu_args, threads, grid, result):
        """Benchmark prologue one kernel execution per PrologueObserver"""

        for obs in self.prologue_observers:
            self.dev.synchronize()
            obs.before_start()
            self.dev.run_kernel(func, gpu_args, threads, grid)
            self.dev.synchronize()
            obs.after_finish()
            result.update(obs.get_results())

    def benchmark_default(self, func, gpu_args, threads, grid, result):
        """Benchmark one kernel execution for 'iterations' at a time"""

        self.dev.synchronize()
        for _ in range(self.iterations):
            for obs in self.benchmark_observers:
                obs.before_start()
            self.dev.synchronize()
            self.dev.start_event()
            self.dev.run_kernel(func, gpu_args, threads, grid)
            self.dev.stop_event()
            for obs in self.benchmark_observers:
                obs.after_start()
            while not self.dev.kernel_finished():
                for obs in self.benchmark_observers:
                    obs.during()
                time.sleep(1e-6)  # one microsecond
            self.dev.synchronize()
            for obs in self.benchmark_observers:
                obs.after_finish()

        for obs in self.benchmark_observers:
            result.update(obs.get_results())


    def benchmark_continuous(self, func, gpu_args, threads, grid, result, duration):
        """Benchmark continuously for at least 'duration' seconds"""
        iterations = int(np.ceil(duration / (result["time"] / 1000)))
        self.dev.synchronize()
        for obs in self.continuous_observers:
            obs.before_start()
        self.dev.start_event()
        for _ in range(iterations):
            self.dev.run_kernel(func, gpu_args, threads, grid)
        self.dev.stop_event()
        for obs in self.continuous_observers:
            obs.after_start()
        while not self.dev.kernel_finished():
            for obs in self.continuous_observers:
                obs.during()
            time.sleep(1e-6)  # one microsecond
        self.dev.synchronize()
        for obs in self.continuous_observers:
            obs.after_finish()

        for obs in self.continuous_observers:
            result.update(obs.get_results())


    def set_nvml_parameters(self, instance):
        """Set the NVML parameters. Avoids setting time leaking into benchmark time."""
        if self.use_nvml:
            if "nvml_pwr_limit" in instance.params:
                new_limit = int(
                    instance.params["nvml_pwr_limit"] * 1000
                )  # user specifies in Watt, but nvml uses milliWatt
                if self.nvml.pwr_limit != new_limit:
                    self.nvml.pwr_limit = new_limit
            if "nvml_gr_clock" in instance.params:
                self.nvml.gr_clock = instance.params["nvml_gr_clock"]
            if "nvml_mem_clock" in instance.params:
                self.nvml.mem_clock = instance.params["nvml_mem_clock"]

        if self.use_tegra:
            if "tegra_gr_clock" in instance.params:
                self.tegra.gr_clock = instance.params["tegra_gr_clock"]


    def benchmark(self, func, gpu_args, instance, verbose, objective, skip_nvml_setting=False):
        """Benchmark the kernel instance."""
        logging.debug("benchmark " + instance.name)
        logging.debug("thread block dimensions x,y,z=%d,%d,%d", *instance.threads)
        logging.debug("grid dimensions x,y,z=%d,%d,%d", *instance.grid)

        if self.use_nvml and not skip_nvml_setting:
            self.set_nvml_parameters(instance)

        # Call the observers to register the configuration to be benchmarked
        for obs in self.dev.observers:
            obs.register_configuration(instance.params)

        result = {}
        try:
            self.benchmark_prologue(func, gpu_args, instance.threads, instance.grid, result)
            self.benchmark_default(func, gpu_args, instance.threads, instance.grid, result)

            if self.continuous_observers:
                duration = 1
                for obs in self.continuous_observers:
                    obs.results = result
                    duration = max(duration, obs.continuous_duration)

                self.benchmark_continuous(
                    func, gpu_args, instance.threads, instance.grid, result, duration
                )

        except Exception as e:
            # some launches may fail because too many registers are required
            # to run the kernel given the current thread block size
            # the desired behavior is to simply skip over this configuration
            # and proceed to try the next one
            skippable_exceptions = [
                "too many resources requested for launch",
                "OUT_OF_RESOURCES",
                "INVALID_WORK_GROUP_SIZE",
            ]
            if any([skip_str in str(e) for skip_str in skippable_exceptions]):
                logging.debug(
                    "benchmark fails due to runtime failure too many resources required"
                )
                if verbose:
                    print(
                        f"skipping config {util.get_instance_string(instance.params)} reason: too many resources requested for launch"
                    )
                result[objective] = util.RuntimeFailedConfig()
            else:
                logging.debug("benchmark encountered runtime failure: " + str(e))
                print("Error while benchmarking:", instance.name)
                raise e
        return result

    def check_kernel_output(
        self, func, gpu_args, instance, answer, atol, verify, verbose
    ):
        """runs the kernel once and checks the result against answer"""
        logging.debug("check_kernel_output")

        #if not using custom verify function, check if the length is the same
        if answer:
            if len(instance.arguments) != len(answer):
                raise TypeError("The length of argument list and provided results do not match.")

            should_sync = [answer[i] is not None for i, arg in enumerate(instance.arguments)]
        else:
            should_sync = [isinstance(arg, (np.ndarray, cp.ndarray, torch.Tensor, DeviceArray)) for arg in instance.arguments]

        # re-copy original contents of output arguments to GPU memory, to overwrite any changes
        # by earlier kernel runs
        for i, arg in enumerate(instance.arguments):
            if should_sync[i]:
                self.dev.memcpy_htod(gpu_args[i], arg)

        # run the kernel
        check = self.run_kernel(func, gpu_args, instance)
        if not check:
            return  # runtime failure occurred that should be ignored, skip correctness check

        # retrieve gpu results to host memory
        result_host = []
        for i, arg in enumerate(instance.arguments):
            if should_sync[i]:
                if isinstance(arg, (np.ndarray, cp.ndarray)):
                    result_host.append(np.zeros_like(arg))
                    self.dev.memcpy_dtoh(result_host[-1], gpu_args[i])
                elif isinstance(arg, torch.Tensor) and isinstance(answer[i], torch.Tensor):
                    if not answer[i].is_cuda:
                        #if the answer is on the host, copy gpu output to host as well
                        result_host.append(torch.zeros_like(answer[i]))
                        self.dev.memcpy_dtoh(result_host[-1], gpu_args[i].tensor)
                    else:
                        result_host.append(gpu_args[i].tensor)
                else:
                    # We should sync this argument, but we do not know how to transfer this type of argument
                    # What do we do? Should we throw an error?
                    result_host.append(None)
            else:
                result_host.append(None)

        # Call the output observers
        for obs in self.output_observers:
            obs.process_output(answer, result_host)

        # There are three scenarios:
        # - if there is a custom verify function, call that.
        # - otherwise, if there are no output observers, call the default verify function
        # - otherwise, the answer is correct (we assume the accuracy observers verified the output)
        if verify:
            correct = verify(answer, result_host, atol=atol)
        elif not self.output_observers:
            correct = _default_verify_function(instance, answer, result_host, atol, verbose)
        else:
            correct = True

        if not correct:
            raise RuntimeError(
                "Kernel result verification failed for: "
                + util.get_config_string(instance.params)
            )

    def compile_and_benchmark(self, kernel_source, gpu_args, params, kernel_options, to):
        # reset previous timers
        last_compilation_time = None
        last_verification_time = None
        last_benchmark_time = None

        verbose = to.verbose
        result = {}

        # Compile and benchmark a kernel instance based on kernel strings and parameters
        instance_string = util.get_instance_string(params)

        logging.debug('compile_and_benchmark ' + instance_string)

        instance = self.create_kernel_instance(kernel_source, kernel_options, params, verbose)
        if isinstance(instance, util.ErrorConfig):
            result[to.objective] = util.InvalidConfig()
        else:
            # Preprocess the argument list. This is required to deal with `MixedPrecisionArray`s
            gpu_args = _preprocess_gpu_arguments(gpu_args, params)

            try:
                # compile the kernel
                start_compilation = time.perf_counter()
                func = self.compile_kernel(instance, verbose)
                if not func:
                    result[to.objective] = util.CompilationFailedConfig()
                else:
                    # add shared memory arguments to compiled module
                    if kernel_options.smem_args is not None:
                        self.dev.copy_shared_memory_args(
                            util.get_smem_args(kernel_options.smem_args, params)
                        )
                    # add constant memory arguments to compiled module
                    if kernel_options.cmem_args is not None:
                        self.dev.copy_constant_memory_args(kernel_options.cmem_args)
                    # add texture memory arguments to compiled module
                    if kernel_options.texmem_args is not None:
                        self.dev.copy_texture_memory_args(kernel_options.texmem_args)

                # stop compilation stopwatch and convert to milliseconds
                last_compilation_time = 1000 * (time.perf_counter() - start_compilation)

                # test kernel for correctness
                if func and (to.answer or to.verify or self.output_observers):
                    start_verification = time.perf_counter()
                    self.check_kernel_output(
                        func, gpu_args, instance, to.answer, to.atol, to.verify, verbose
                    )
                    last_verification_time = 1000 * (
                        time.perf_counter() - start_verification
                    )

                # benchmark
                if func:
                    # setting the NVML parameters here avoids this time from leaking into the benchmark time, ends up in framework time instead
                    if self.use_nvml:
                        self.set_nvml_parameters(instance)
                    start_benchmark = time.perf_counter()
                    result.update(
                        self.benchmark(func, gpu_args, instance, verbose, to.objective, skip_nvml_setting=False)
                    )
                    last_benchmark_time = 1000 * (time.perf_counter() - start_benchmark)

            except Exception as e:
                # dump kernel sources to temp file
                temp_filenames = instance.prepare_temp_files_for_error_msg()
                print(
                    "Error while compiling or benchmarking, see source files: "
                    + " ".join(temp_filenames)
                )
                raise e

            # clean up any temporary files, if no error occurred
            instance.delete_temp_files()

        result["compile_time"] = last_compilation_time or 0
        result["verification_time"] = last_verification_time or 0
        result["benchmark_time"] = last_benchmark_time or 0

        return result

    def compile_kernel(self, instance, verbose):
        """compile the kernel for this specific instance"""
        logging.debug("compile_kernel " + instance.name)

        # compile kernel_string into device func
        func = None
        try:
            func = self.dev.compile(instance)
        except Exception as e:
            # compiles may fail because certain kernel configurations use too
            # much shared memory for example, the desired behavior is to simply
            # skip over this configuration and try the next one
            shared_mem_error_messages = [
                "uses too much shared data",
                "local memory limit exceeded",
                r"local memory \(\d+\) exceeds limit \(\d+\)",
            ]
            error_message = str(e.stderr) if hasattr(e, "stderr") else str(e)
            if any(re.search(msg, error_message) for msg in shared_mem_error_messages):
                logging.debug(
                    "compile_kernel failed due to kernel using too much shared memory"
                )
                if verbose:
                    print(
                        f"skipping config {util.get_instance_string(instance.params)} reason: too much shared memory used"
                    )
            else:
                print("compile_kernel failed due to error: " + error_message)
                print("Error while compiling:", instance.name)
                raise e
        return func

    @staticmethod
    def preprocess_gpu_arguments(old_arguments, params):
        """ Get a flat list of arguments based on the configuration given by `params` """
        return _preprocess_gpu_arguments(old_arguments, params)

    def copy_shared_memory_args(self, smem_args):
        """adds shared memory arguments to the most recently compiled module"""
        self.dev.copy_shared_memory_args(smem_args)

    def copy_constant_memory_args(self, cmem_args):
        """adds constant memory arguments to the most recently compiled module"""
        self.dev.copy_constant_memory_args(cmem_args)

    def copy_texture_memory_args(self, texmem_args):
        """adds texture memory arguments to the most recently compiled module"""
        self.dev.copy_texture_memory_args(texmem_args)

    def create_kernel_instance(self, kernel_source, kernel_options, params, verbose):
        """create kernel instance from kernel source, parameters, problem size, grid divisors, and so on"""
        grid_div = (
            kernel_options.grid_div_x,
            kernel_options.grid_div_y,
            kernel_options.grid_div_z,
        )

        # insert default block_size_names if needed
        if not kernel_options.block_size_names:
            kernel_options.block_size_names = util.default_block_size_names

        # setup thread block and grid dimensions
        threads, grid = util.setup_block_and_grid(
            kernel_options.problem_size,
            grid_div,
            params,
            kernel_options.block_size_names,
        )
        if np.prod(threads) > self.dev.max_threads:
            if verbose:
                print(
                    f"skipping config {util.get_instance_string(params)} reason: too many threads per block"
                )
            return util.InvalidConfig()

        # obtain the kernel_string and prepare additional files, if any
        name, kernel_string, temp_files = kernel_source.prepare_list_of_files(
            kernel_options.kernel_name,
            params,
            grid,
            threads,
            kernel_options.block_size_names,
        )

        # check for templated kernel
        if kernel_source.lang in ["CUDA", "NVCUDA", "HIP"] and "<" in name and ">" in name:
            kernel_string, name = wrap_templated_kernel(kernel_string, name)

        # Preprocess GPU arguments. Require for handling `Tunable` arguments
        arguments = _preprocess_gpu_arguments(kernel_options.arguments, params)

        #collect everything we know about this instance and return it
        return KernelInstance(name, kernel_source, kernel_string, temp_files, threads, grid, params, arguments)

    def get_environment(self):
        """Return dictionary with information about the environment"""
        return self.dev.env

    def memcpy_dtoh(self, dest, src):
        """perform a device to host memory copy"""
        self.dev.memcpy_dtoh(dest, src)

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the kernel, allocates gpu mem if necessary"""
        flat_args = []

        # Flatten all arguments into a single list. Required to deal with `Tunable`s
        for argument in arguments:
            if isinstance(argument, Tunable):
                flat_args.extend(argument.values())
            else:
                flat_args.append(argument)

        flat_gpu_args = iter(self.dev.ready_argument_list(flat_args))

        # Unflatten the arguments back into arrays.
        gpu_args = []
        for argument in arguments:
            if isinstance(argument, Tunable):
                arrays = dict()
                for key in argument:
                    arrays[key] = next(flat_gpu_args)

                gpu_args.append(Tunable(argument.param_key, arrays))
            else:
                gpu_args.append(next(flat_gpu_args))

        return gpu_args

    def run_kernel(self, func, gpu_args, instance):
        """Run a compiled kernel instance on a device"""
        logging.debug("run_kernel %s", instance.name)
        logging.debug("thread block dims (%d, %d, %d)", *instance.threads)
        logging.debug("grid dims (%d, %d, %d)", *instance.grid)

        try:
            self.dev.run_kernel(func, gpu_args, instance.threads, instance.grid)
        except Exception as e:
            if "too many resources requested for launch" in str(
                e
            ) or "OUT_OF_RESOURCES" in str(e):
                logging.debug(
                    "ignoring runtime failure due to too many resources required"
                )
                return False
            else:
                logging.debug("encountered unexpected runtime failure: " + str(e))
                raise e
        return True


def _preprocess_gpu_arguments(old_arguments, params):
    """ Get a flat list of arguments based on the configuration given by `params` """
    new_arguments = []

    for argument in old_arguments:
        if isinstance(argument, Tunable):
            new_arguments.append(argument.select_for_configuration(params))
        else:
            new_arguments.append(argument)

    return new_arguments


def _default_verify_function(instance, answer, result_host, atol, verbose):
    """default verify function based on np.allclose"""

    # first check if the length is the same
    if len(instance.arguments) != len(answer):
        raise TypeError(
            "The length of argument list and provided results do not match."
        )
    # for each element in the argument list, check if the types match
    for i, arg in enumerate(instance.arguments):
        if answer[i] is not None:  # skip None elements in the answer list
            if isinstance(answer[i], (np.ndarray, cp.ndarray)) and isinstance(
                arg, (np.ndarray, cp.ndarray)
            ):
                if not np.can_cast(arg.dtype, answer[i].dtype):
                    raise TypeError(
                        f"Element {i} of the expected results list has a dtype that is not compatible with the dtype of the kernel output: "
                        + str(answer[i].dtype)
                        + " != "
                        + str(arg.dtype)
                        + "."
                    )
                if answer[i].size != arg.size:
                    raise TypeError(
                        f"Element {i} of the expected results list has a size different from "
                        + "the kernel argument: "
                        + str(answer[i].size)
                        + " != "
                        + str(arg.size)
                        + "."
                    )
            elif isinstance(answer[i], torch.Tensor) and isinstance(arg, torch.Tensor):
                if answer[i].dtype != arg.dtype:
                    raise TypeError(
                        f"Element {i} of the expected results list is not of the same dtype as the kernel output: "
                        + str(answer[i].dtype)
                        + " != "
                        + str(arg.dtype)
                        + "."
                    )
                if answer[i].size() != arg.size():
                    raise TypeError(
                        f"Element {i} of the expected results list has a size different from "
                        + "the kernel argument: "
                        + str(answer[i].size)
                        + " != "
                        + str(arg.size)
                        + "."
                    )

            elif isinstance(answer[i], np.number) and isinstance(arg, np.number):
                if answer[i].dtype != arg.dtype:
                    raise TypeError(
                        f"Element {i} of the expected results list is not the same as the kernel output: "
                        + str(answer[i].dtype)
                        + " != "
                        + str(arg.dtype)
                        + "."
                    )
            else:
                # either answer[i] and argument have different types or answer[i] is not a numpy type
                if not isinstance(
                    answer[i], (np.ndarray, cp.ndarray, torch.Tensor)
                ) or not isinstance(answer[i], np.number):
                    raise TypeError(
                        f"Element {i} of expected results list is not a numpy/cupy ndarray, torch Tensor or numpy scalar."
                    )
                else:
                    raise TypeError(
                        f"Element {i} of expected results list and kernel arguments have different types."
                    )

    def _ravel(a):
        if hasattr(a, "ravel") and len(a.shape) > 1:
            return a.ravel()
        return a

    def _flatten(a):
        if hasattr(a, "flatten"):
            return a.flatten()
        return a

    correct = True
    for i, arg in enumerate(instance.arguments):
        expected = answer[i]
        if expected is not None:
            result = _ravel(result_host[i])
            expected = _flatten(expected)
            if any([isinstance(array, cp.ndarray) for array in [expected, result]]):
                output_test = cp.allclose(expected, result, atol=atol)
            elif isinstance(expected, torch.Tensor) and isinstance(
                result, torch.Tensor
            ):
                output_test = torch.allclose(expected, result, atol=atol)
            else:
                output_test = np.allclose(expected, result, atol=atol)

            if not output_test and verbose:
                print(
                    "Error: "
                    + util.get_config_string(instance.params)
                    + " detected during correctness check"
                )
                print(
                    "this error occurred when checking value of the %oth kernel argument"
                    % (i,)
                )
                print(
                    "Printing kernel output and expected result, set verbose=False to suppress this debug print"
                )
                np.set_printoptions(edgeitems=50)
                print("Kernel output:")
                print(result)
                print("Expected:")
                print(expected)
            correct = correct and output_test

    if not correct:
        logging.debug("correctness check has found a correctness issue")

    return correct


# these functions facilitate compiling templated kernels with PyCuda
def split_argument_list(argument_list):
    """split all arguments in a list into types and names"""
    regex = r"(.*[\s*]+)(.+)?"
    type_list = []
    name_list = []
    for arg in argument_list:
        match = re.match(regex, arg, re.S)
        if not match:
            raise ValueError("error parsing templated kernel argument list")
        type_list.append(re.sub(r"\s+", " ", match.group(1).strip(), flags=re.S))
        name_list.append(match.group(2).strip())
    return type_list, name_list


def apply_template_typenames(type_list, templated_typenames):
    """replace the typename tokens in type_list with their templated typenames"""

    def replace_typename_token(matchobj):
        """function for a whitespace preserving token regex replace"""
        # replace only the match, leaving the whitespace around it as is
        return (
            matchobj.group(1)
            + templated_typenames[matchobj.group(2)]
            + matchobj.group(3)
        )

    for i, arg_type in enumerate(type_list):
        for k, v in templated_typenames.items():
            # if the templated typename occurs as a token in the string, meaning that it is enclosed in
            # beginning of string or whitespace, and end of string, whitespace or star
            regex = r"(^|\s+)(" + k + r")($|\s+|\*)"
            sub = re.sub(regex, replace_typename_token, arg_type, flags=re.S)
            type_list[i] = sub


def get_templated_typenames(template_parameters, template_arguments):
    """based on the template parameters and arguments, create dict with templated typenames"""
    templated_typenames = {}
    for i, param in enumerate(template_parameters):
        if "typename " in param:
            typename = param[9:]
            templated_typenames[typename] = template_arguments[i]
    return templated_typenames


def wrap_templated_kernel(kernel_string, kernel_name):
    """rewrite kernel_string to insert wrapper function for templated kernel"""
    # parse kernel_name to find template_arguments and real kernel name
    name = kernel_name.split("<")[0]
    template_arguments = re.search(r".*?<(.*)>", kernel_name, re.S).group(1).split(",")

    # parse templated kernel definition
    # relatively strict regex that does not allow nested template parameters like vector<TF>
    # within the template parameter list
    regex = (
        r"template\s*<([^>]*?)>\s*__global__\s+void\s+(__launch_bounds__\([^\)]+?\)\s+)?"
        + name
        + r"\s*\((.*?)\)\s*\{"
    )
    match = re.search(regex, kernel_string, re.S)
    if not match:
        raise ValueError("could not find templated kernel definition")

    template_parameters = match.group(1).split(",")
    argument_list = match.group(3).split(",")
    argument_list = [
        s.strip() for s in argument_list
    ]  # remove extra whitespace around 'type name' strings

    type_list, name_list = split_argument_list(argument_list)

    templated_typenames = get_templated_typenames(
        template_parameters, template_arguments
    )
    apply_template_typenames(type_list, templated_typenames)

    # replace __global__ with __device__ in the templated kernel definition
    # could do a more precise replace, but __global__ cannot be used elsewhere in the definition
    definition = match.group(0).replace("__global__", "__device__")

    # there is a __launch_bounds__() group that is matched
    launch_bounds = ""
    if match.group(2):
        definition = definition.replace(match.group(2), " ")
        launch_bounds = match.group(2)

    # generate code for the compile-time template instantiation
    template_instantiation = (
        f"template __device__ void {kernel_name}(" + ", ".join(type_list) + ");\n"
    )

    # generate code for the wrapper kernel
    new_arg_list = ", ".join([" ".join((a, b)) for a, b in zip(type_list, name_list)])
    wrapper_function = (
        '\nextern "C" __global__ void '
        + launch_bounds
        + name
        + "_wrapper("
        + new_arg_list
        + ") {\n  "
        + kernel_name
        + "("
        + ", ".join(name_list)
        + ");\n}\n"
    )

    # copy kernel_string, replace definition and append template instantiation and wrapper function
    new_kernel_string = kernel_string[:]
    new_kernel_string = new_kernel_string.replace(match.group(0), definition)
    new_kernel_string += "\n" + template_instantiation
    new_kernel_string += wrapper_function

    return new_kernel_string, name + "_wrapper"
