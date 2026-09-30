"""Module for grouping the core functionality needed by most runners."""

import logging
import re
import time
from collections import namedtuple
from warnings import warn

import numpy as np


def _get_cupy():
    try:
        import cupy as _cp
    except ImportError:
        return None
    return _cp


import kernel_tuner.util as util
from kernel_tuner.accuracy import Tunable
from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.kernel_sources.kernel_source import KernelSource  # noqa: F401 (re-exported as core.KernelSource)
from kernel_tuner.observers.observer import BenchmarkObserver, ContinuousObserver, OutputObserver, PrologueObserver
from kernel_tuner.observers.tegra import TegraObserver

try:
    import torch
except ImportError:
    torch = util.TorchPlaceHolder()

try:
    from hip._util.types import DeviceArray
except ImportError:
    DeviceArray = Exception  # using Exception here as a type that will never be among kernel arguments


_KernelInstance = namedtuple(
    "_KernelInstance",
    [
        "name",
        "kernel_source",
        "kernel_string",
        "kernel_fn",
        "temp_files",
        "threads",
        "grid",
        "params",
        "arguments",
    ],
)


class KernelInstance(_KernelInstance):
    """Class that represents the specific parameterized instance of a kernel."""

    def __new__(cls, *args, **kwargs):
        # Detect old-style calls (without kernel_fn and gpu_kwargs) for old tests
        if len(args) == 8:  # old version
            name, kernel_source, kernel_string, temp_files, threads, grid, params, arguments = args
            kernel_fn = None
            args = (name, kernel_source, kernel_string, kernel_fn, temp_files, threads, grid, params, arguments)
        elif "kernel_fn" not in kwargs and len(args) < 9:
            kwargs["kernel_fn"] = None
        return super().__new__(cls, *args, **kwargs)
    
    def delete_temp_files(self):
        """Delete any generated temp files."""
        tmp_files_list = self.temp_files.values() if isinstance(self.temp_files, dict) else self.temp_files
        for tmp_file in tmp_files_list:
            util.delete_temp_file(tmp_file)

    def prepare_temp_files_for_error_msg(self):
        """Prepare temp file with source code, and return list of temp file names."""
        if type(self.kernel_source).__name__ == "KernelSourceFn":
            return [] # already done during compilation

        temp_filename = util.get_temp_filename(suffix=self.kernel_source.get_suffix())
        util.write_file(temp_filename, self.kernel_string)
        ret = [temp_filename]
        ret.extend(self.temp_files.values())
        return ret


# KernelSource has been moved to kernel_sources/kernel_source_str.py to support Generic Python



def instantiate_observer(observer, args):
    """Instantiate or build an observer from a class/factory/instance."""
    if isinstance(observer, BenchmarkObserver):
        return observer
    elif callable(observer):
        # Check again if BenchmarkObserver
        return instantiate_observer(observer(args), args)
    else:
        raise TypeError(f"Invalid observer: {observer!r} does not extend BenchmarkObserver")


def _select_default_cuda_backend():
    """Select default CUDA backend, looks for which backends are installed."""
    # First try cuda-python (nvcuda)
    from kernel_tuner.backends.nvcuda import CudaFunctions, driver

    if driver:
        return CudaFunctions
    # Then try Cupy
    if _get_cupy():
        from kernel_tuner.backends.cupy import CupyFunctions

        return CupyFunctions
    # Then try PyCUDA
    from kernel_tuner.backends.pycuda import PyCudaFunctions, pycuda_available

    if pycuda_available:
        return PyCudaFunctions
    # Ran out of options
    raise RuntimeError(
        "Error: CUDA selected/detected, but missing CUDA dependencies, please run 'pip install cuda-python', or install cupy or pycuda."
    )


class DeviceInterface(object):
    """Class that offers a High-Level Device Interface to the rest of the Kernel Tuner."""

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
        """Instantiate the DeviceInterface, based on language in kernel source.

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
        self.requires_warmup = True

        logging.debug("DeviceInterface instantiated, lang=%s", lang)

        
        # Ensure observers is a list
        observers = observers or []

        # Observers can either be an object that extends BenchmarkObserver or
        # lambda function that returns such an object.
        observer_args = dict(device=device, platform=platform, compiler=compiler, lang=lang)
        observers = [instantiate_observer(ob, observer_args) for ob in observers]

        backend_options = dict(compiler_options=compiler_options, iterations=iterations)

        # first check for explicitly selected backends
        if lang.upper() == "PYCUDA":
            from kernel_tuner.backends.pycuda import PyCudaFunctions

            backend = PyCudaFunctions
        elif lang.upper() == "CUPY":
            from kernel_tuner.backends.cupy import CupyFunctions

            backend = CupyFunctions
        elif lang.upper() == "NVCUDA":
            from kernel_tuner.backends.nvcuda import CudaFunctions

            backend = CudaFunctions
        elif lang.upper() == "CUDA":
            # Select default CUDA backend, based on availability
            backend = _select_default_cuda_backend()
        elif lang.upper() == "OPENCL":
            from kernel_tuner.backends.opencl import OpenCLFunctions

            backend = OpenCLFunctions
            backend_options["platform"] = platform
        elif lang.upper() == "HIP":
            from kernel_tuner.backends.hip import HipFunctions

            backend = HipFunctions
        elif lang.upper() == "JULIA":
            from kernel_tuner.backends.julia import JuliaFunctions

            backend = JuliaFunctions
        elif lang.upper() == "HYPERTUNER":
            from kernel_tuner.backends.hypertuner import HypertunerFunctions

            backend = HypertunerFunctions
            self.requires_warmup = False
        elif lang.upper() in ["C", "FORTRAN"]:
            from kernel_tuner.backends.compiler import CompilerFunctions

            backend = CompilerFunctions
            backend_options["compiler"] = compiler
            backend_options["observers"] = observers
        elif lang.upper() == "GENERIC_PYTHON":
            from kernel_tuner.backends.generic_python import GenericPythonFunctions

            backend = GenericPythonFunctions
        else:
            raise NotImplementedError(
                "Sorry, support for languages other than CUDA, OpenCL, HIP, C, Julia, Fortran and Generic Python is not implemented yet"
            )

        if issubclass(backend, GPUBackend):
            backend_options["device"] = device
            backend_options["observers"] = observers
        self.dev = backend(**backend_options)

        # look for NVMLObserver and TegraObserver in observers, if present, enable special tunable parameters through nvml/tegra
        self.use_nvml = False
        self.use_tegra = False
        self.continuous_observers = []
        self.output_observers = []
        self.prologue_observers = []
        if observers:
            try:
                from kernel_tuner.observers.nvml import NVMLObserver as _NVMLObserver
            except ImportError:
                _NVMLObserver = None
            for obs in observers:
                if _NVMLObserver is not None and isinstance(obs, _NVMLObserver):
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

        # for JULIA, add the JIT warmup prologue observer
        if lang.upper() == "JULIA":
            from kernel_tuner.observers.julia import JuliaJITWarmup

            self.prologue_observers.append(JuliaJITWarmup(self.dev.backend))
            self.prologue_observers.append(JuliaJITWarmup(self.dev.backend))

        # Take list of observers from self.dev because Backends tend to add their own observer
        self.benchmark_observers = [
            obs for obs in self.dev.observers if not isinstance(obs, (ContinuousObserver, PrologueObserver))
        ]

        self.iterations = iterations

        self.lang = lang
        self.units = self.dev.units
        self.name = self.dev.name
        self.max_threads = self.dev.max_threads
        if not quiet:
            print("Using: " + self.dev.name)

    def run_kernel_bench(self, func, gpu_args, threads, grid, stream=None, params=None):
        if self.lang.upper() == "GENERIC_PYTHON":  # Generic Python needs params to launch
            self.dev.run_kernel(func, gpu_args, threads, grid, params=params)
        else:
            self.dev.run_kernel(func, gpu_args, threads, grid)

    def benchmark_prologue(self, func, gpu_args, threads, grid, result, params=None):
        """Benchmark prologue one kernel execution per PrologueObserver."""
        for obs in self.prologue_observers:
            self.dev.synchronize()
            obs.before_start()
            self.run_kernel_bench(func, gpu_args, threads, grid, stream=None, params=params)
            self.dev.synchronize()
            obs.after_finish()
            result.update(obs.get_results())

    def benchmark_default(self, func, gpu_args, threads, grid, result, params=None):
        """Benchmark one kernel execution for 'iterations' at a time."""
        self.dev.synchronize()
        for _ in range(self.iterations):
            for obs in self.benchmark_observers:
                obs.before_start()
            self.dev.synchronize()
            self.dev.start_event()
            self.run_kernel_bench(func, gpu_args, threads, grid, stream=None, params=params)
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

            #time.sleep(0.1) # prevent termal throttling

        for obs in self.benchmark_observers:
            result.update(obs.get_results())

    def benchmark_continuous(self, func, gpu_args, threads, grid, result, duration, params=None):
        """Benchmark continuously for at least 'duration' seconds."""
        iterations = int(np.ceil(duration / (result["time"] / 1000)))
        self.dev.synchronize()
        for obs in self.continuous_observers:
            obs.before_start()
        self.dev.start_event()
        for _ in range(iterations):
            self.run_kernel_bench(func, gpu_args, threads, grid, stream=None, params=params)
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

        # Set execution parameters
        if self.use_nvml and not skip_nvml_setting:
            self.set_nvml_parameters(instance)
        if "cuda_sm_percentage" in instance.params:
            # Currently only supported on cuda-python (NVCUDA)
            self.dev.set_sm_percentage(instance.params["cuda_sm_percentage"])

        # Call the observers to register the configuration to be benchmarked
        for obs in self.dev.observers:
            obs.register_configuration(instance.params)

        result = {}
        try:
            self.benchmark_prologue(func, gpu_args, instance.threads, instance.grid, result, instance.params)
            self.benchmark_default(func, gpu_args, instance.threads, instance.grid, result, instance.params)

            duration = 1
            for obs in self.continuous_observers:
                obs.results = result
                duration = max(duration, obs.continuous_duration)
            if len(self.continuous_observers) > 0:
                self.benchmark_continuous(func, gpu_args, instance.threads, instance.grid, result, duration, instance.params)

        except Exception as e:
            # some launches may fail because too many registers are required
            # to run the kernel given the current thread block size
            # the desired behavior is to simply skip over this configuration
            # and proceed to try the next one
            skippable_exceptions = [
                "too many resources requested for launch",
                "OUT_OF_RESOURCES",
                "INVALID_WORK_GROUP_SIZE",
                "a bounds error was thrown during kernel execution",
                "Julia kernel launch failed",
            ]
            if any([skip_str in str(e) for skip_str in skippable_exceptions]):
                logging.debug("benchmark fails due to runtime failure / too many resources required")
                if "julia" in str(e).lower() and verbose:
                    warn(
                        f"skipping config {util.get_instance_string(instance.params)} reason: Julia kernel launch failed because of:\n{e}"
                    )
                elif verbose:
                    print(
                        f"skipping config {util.get_instance_string(instance.params)} reason: too many resources requested for launch"
                    )
                result["__error__"] = util.RuntimeFailedConfig()
            else:
                logging.debug("benchmark encountered runtime failure: " + str(e))
                print("Error while benchmarking:", instance.name)
                raise e

        assert util.check_result_type(result), "The error in a result MUST be an actual error."

        return result

    def check_kernel_output(self, func, gpu_args, instance, answer, atol, verify, verbose):
        """Runs the kernel once and checks the result against answer."""
        logging.debug("check_kernel_output")
        cp = _get_cupy()

        # get the answer for this parameter configuration
        if isinstance(answer, Tunable):
            answer = answer.select_for_configuration(instance.params)

        # convert juliacall arrays to numpy arrays where necessary
        if answer is not None:
            answer = [np.array(a) if util.is_julia_array(a) else a for a in util.possible_julia_vector_to_list(answer)]
        for i, arg in enumerate(instance.arguments):
            if util.is_julia_array(arg) and isinstance(answer[i], np.ndarray):
                instance.arguments[i] = np.array(arg, dtype=answer[i].dtype)

        # if not using custom verify function, check if the length is the same
        if answer:
            if len(instance.arguments) != len(answer):
                raise TypeError("The length of argument list and provided results do not match.")

            # for Julia arrays, we always want to sync
            should_sync = [
                answer[i] is not None or util.is_julia_array(arg) for i, arg in enumerate(instance.arguments)
            ]
        else:
            cupy_ndarray = (cp.ndarray,) if cp is not None else ()
            should_sync = [
                isinstance(arg, (np.ndarray, cp.ndarray, torch.Tensor, DeviceArray) + cupy_ndarray)
                or util.is_julia_array(arg)
                for i, arg in enumerate(instance.arguments)
            ]

        # re-copy original contents of output arguments to GPU memory, to overwrite any changes
        # by earlier kernel runs
        self.dev.refresh_memory(gpu_args, instance.arguments, should_sync)

        # run the kernel
        self.dev.synchronize()
        check = self.run_kernel(func, gpu_args, instance)
        self.dev.synchronize()
        if not check:
            # runtime failure occurred that should be ignored, skip correctness check
            return

        # retrieve gpu results to host memory
        result_host = self.retrieve_results_to_host(instance.arguments, should_sync, gpu_args, answer)

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
            print("expected: ", answer, "\ngot: ", result_host)
            raise RuntimeError("Kernel result verification failed for: " + util.get_config_string(instance.params))

    def compile_and_benchmark(self, kernel_source, gpu_args, params, kernel_options, to):
        # reset previous timers
        last_compilation_time = None
        last_verification_time = None
        last_benchmark_time = None

        verbose = to.verbose
        result = {}

        # Compile and benchmark a kernel instance based on kernel strings and parameters
        instance_string = util.get_instance_string(params)

        logging.debug("compile_and_benchmark " + instance_string)
        instance = self.create_kernel_instance(kernel_source, kernel_options, params, verbose)
        if isinstance(instance, util.ErrorConfig):
            result["__error__"] = util.InvalidConfig()
        else:
            # Preprocess the argument list. This is required to deal with `MixedPrecisionArray`s
            gpu_args = _preprocess_gpu_arguments(gpu_args, params)

            try:
                # compile the kernel
                start_compilation = time.perf_counter()
                func = self.compile_kernel(instance, verbose, gpu_args)
                if not func:
                    result["__error__"] = util.CompilationFailedConfig()
                else:
                    # add shared memory arguments to compiled module
                    if kernel_options.smem_args is not None:
                        self.dev.copy_shared_memory_args(util.get_smem_args(kernel_options.smem_args, params))
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
                    self.check_kernel_output(func, gpu_args, instance, to.answer, to.atol, to.verify, verbose)
                    last_verification_time = 1000 * (time.perf_counter() - start_verification)

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
                print("Error while compiling or benchmarking, see source files: " + " ".join(temp_filenames))
                raise e

            # clean up any temporary files, if no error occurred
            instance.delete_temp_files()

        # For Python DSLs, the compilation time also includes one kernel run, so we subtract the runtime
        if self.lang.upper() == "GENERIC_PYTHON" and last_compilation_time and "time" in result:
            last_compilation_time -= result["time"]

        result["compile_time"] = last_compilation_time or 0
        result["verification_time"] = last_verification_time or 0
        result["benchmark_time"] = last_benchmark_time or 0

        assert util.check_result_type(result), "The error in a result MUST be an actual error."

        return result

    def compile_kernel(self, instance, verbose, gpu_args=None):
        """Compile the kernel for this specific instance.

        Generic Python kernels are compiled by running them once, which requires ``gpu_args``.
        """
        logging.debug("compile_kernel " + instance.name)

        # compile kernel_string into device func
        func = None
        try:
            if self.lang.upper() == "GENERIC_PYTHON":
                func = self.dev.compile(instance, gpu_args)
            else:
                func = self.dev.compile(instance)
        except Exception as e:
            # compiles may fail because certain kernel configurations use too
            # much shared memory for example, the desired behavior is to simply
            # skip over this configuration and try the next one
            error_message = str(e.stderr) if hasattr(e, "stderr") else str(e)
            if self.lang.upper() == "GENERIC_PYTHON":
                # Python DSLs raise many different errors, the backend classifies them.
                # Only skip configurations that are recognized as using too many resources.
                skippable = self.dev.classify_compile_exception(e) == "resource_error"
                reason = f"\n{e}"
            else:
                shared_mem_error_messages = [
                    "uses too much shared data",
                    "local memory limit exceeded",
                    r"local memory \(\d+\) exceeds limit \(\d+\)",
                ]
                skippable = any(re.search(msg, error_message) for msg in shared_mem_error_messages)
                reason = "too much shared memory used"
            if skippable:
                logging.debug("compile_kernel failed due to kernel using too many resources")
                if verbose:
                    print(f"skipping config {util.get_instance_string(instance.params)} reason: {reason}")
            else:
                print("compile_kernel failed due to error: " + error_message)
                print("Error while compiling:", instance.name)
                raise e
        return func

    @staticmethod
    def preprocess_gpu_arguments(old_arguments, params):
        """Get a flat list of arguments based on the configuration given by `params`."""
        return _preprocess_gpu_arguments(old_arguments, params)

    def copy_shared_memory_args(self, smem_args):
        """Adds shared memory arguments to the most recently compiled module."""
        self.dev.copy_shared_memory_args(smem_args)

    def copy_constant_memory_args(self, cmem_args):
        """Adds constant memory arguments to the most recently compiled module."""
        self.dev.copy_constant_memory_args(cmem_args)

    def copy_texture_memory_args(self, texmem_args):
        """Adds texture memory arguments to the most recently compiled module."""
        self.dev.copy_texture_memory_args(texmem_args)

    def create_kernel_instance(self, kernel_source, kernel_options, params, verbose):
        """Create kernel instance from kernel source, parameters, problem size, grid divisors, and so on."""
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

        if kernel_source.lang.upper() != "GENERIC_PYTHON" and np.prod(threads) > self.dev.max_threads:
            if verbose:
                print(f"skipping config {util.get_instance_string(params)} reason: too many threads per block")
            return util.InvalidConfig()

        # obtain the kernel_string and prepare additional files, if any
        instance_data = kernel_source.prepare_kernel_instance(
            kernel_options,
            params,
            grid,
            threads,
        )

        # templated CUDA kernels are wrapped by KernelSourceStr.prepare_kernel_instance
        name = instance_data.kernel_name
        kernel_string = instance_data.kernel_str

        # Preprocess GPU arguments. Require for handling `Tunable` arguments
        arguments = _preprocess_gpu_arguments(kernel_options.arguments, params)

        # collect everything we know about this instance and return it
        return KernelInstance(
            name=name,
            kernel_source=kernel_source,
            kernel_string=kernel_string,
            kernel_fn=instance_data.kernel_fn,
            temp_files=instance_data.temp_files,
            threads=threads,
            grid=grid,
            params=params,
            arguments=arguments,
        )

    def get_environment(self):
        """Return dictionary with information about the environment."""
        return self.dev.env

    def memcpy_dtoh(self, dest, src):
        """Perform a device to host memory copy."""
        self.dev.memcpy_dtoh(dest, src)

    def ready_argument_list(self, arguments):
        """Ready argument list to be passed to the kernel, allocates gpu mem if necessary."""
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
        """Run a compiled kernel instance on a device."""
        logging.debug("run_kernel %s", instance.name)
        logging.debug("thread block dims (%d, %d, %d)", *instance.threads)
        logging.debug("grid dims (%d, %d, %d)", *instance.grid)

        try:
            self.run_kernel_bench(func, gpu_args, instance.threads, instance.grid, stream=None, params=instance.params)
        except Exception as e:
            if "too many resources requested for launch" in str(e) or "OUT_OF_RESOURCES" in str(e):
                logging.debug("ignoring runtime failure due to too many resources required")
                return False
            else:
                logging.debug("encountered unexpected runtime failure: " + str(e))
                raise e
        return True
    

    def retrieve_results_to_host(self, arguments: list, should_sync: list[bool], gpu_args, answer: list):
        """Retrieve results from device to host memory for all arguments that should be synchronized."""
        result_host = []
        if self.lang.upper() == "GENERIC_PYTHON":
            # arguments are either Torch tensors or built-in Python types
            for i, arg in enumerate(gpu_args):
                if not should_sync[i]:
                    result_host.append(None)
                elif isinstance(arg, torch.Tensor):
                    result_host.append(arg.cpu())
                else:
                    result_host.append(arg)
            return result_host
        for i, arg in enumerate(arguments):
            if not should_sync[i]:
                result_host.append(None)
                continue
            cp = _get_cupy()
            cupy_ndarray = (cp.ndarray,) if cp is not None else ()
            if isinstance(arg, (np.ndarray,) + cupy_ndarray) or util.is_julia_array(arg):
                result_host.append(np.zeros_like(arg))
                self.dev.memcpy_dtoh(result_host[-1], gpu_args[i])
            elif isinstance(arg, torch.Tensor):
                result = torch.empty(arg.shape, dtype=arg.dtype, device="cpu")
                self.dev.memcpy_dtoh(result, gpu_args[i])
                # verification compares on the device of the answer
                if answer is not None and isinstance(answer[i], torch.Tensor) and answer[i].is_cuda:
                    result = result.to(answer[i].device)
                result_host.append(result)
            else:
                # We should sync this argument, but we do not know how to transfer this type of argument
                # What do we do? Should we throw an error?
                warn(f"Argument {i} is of type {type(arg)} and should be synchronized, but is not implemented.")
                result_host.append(None)
        return result_host


def _preprocess_gpu_arguments(old_arguments, params):
    """Get a flat list of arguments based on the configuration given by `params`."""
    new_arguments = []

    for argument in old_arguments:
        if isinstance(argument, Tunable):
            new_arguments.append(argument.select_for_configuration(params))
        else:
            new_arguments.append(argument)

    return new_arguments


def _default_verify_function(instance, answer, result_host, atol, verbose):
    """Default verify function based on np.allclose."""
    # first check if the length is the same
    if len(instance.arguments) != len(answer):
        raise TypeError("The length of argument list and provided results do not match.")
    # for each element in the argument list, check if the types match
    for i, arg in enumerate(instance.arguments):
        if answer[i] is not None:  # skip None elements in the answer list
            # convert Julia Arrays to numpy arrays for verification
            if util.is_julia_array(arg):
                arg = np.array(arg)
            if util.is_julia_array(answer[i]):
                answer[i] = np.array(answer[i])

            cp = _get_cupy()
            cupy_ndarray = (cp.ndarray,) if cp is not None else ()
            if isinstance(answer[i], (np.ndarray,) + cupy_ndarray) and isinstance(arg, (np.ndarray,) + cupy_ndarray):
                if not np.can_cast(arg.dtype, answer[i].dtype):
                    raise TypeError(
                        f"Element {i} of the expected results list has a dtype that is not compatible with the dtype of the kernel output: "
                        + str(answer[i].dtype)
                        + " != "
                        + str(arg.dtype)
                        + "."
                    )
                if answer[i].size != arg.size:
                    raise ValueError(
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
                    raise ValueError(
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
                cp = _get_cupy()
                cupy_ndarray = (cp.ndarray,) if cp is not None else ()
                if not isinstance(answer[i], (np.ndarray, torch.Tensor) + cupy_ndarray) or not isinstance(
                    answer[i], np.number
                ):
                    raise TypeError(
                        f"Arg or Element {i} of expected results list is {type(arg)} / {type(answer[i])}, not a numpy/cupy ndarray, torch Tensor or numpy scalar."  # noqa: E501
                    )
                else:
                    raise TypeError(f"Element {i} of expected results list and kernel arguments have different types.")

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
            cp = _get_cupy()
            has_cp_array = False if not cp else any([isinstance(array, cp.ndarray) for array in [expected, result]])
            lib = (
                cp
                if has_cp_array
                else torch
                if isinstance(expected, torch.Tensor) and isinstance(result, torch.Tensor)
                else np
            )
            expected_nan = lib.isnan(expected)
            output_test = lib.allclose(expected, result, atol=atol, equal_nan=bool(expected_nan.any()))
            if expected_nan.any():
                warn(
                    f"Answer contains {expected_nan.sum()} NaNs. NaN values will now be considered equal in comparison."
                )

            if not output_test and verbose:
                print("Error: " + util.get_config_string(instance.params) + " detected during correctness check")
                print("this error occurred when checking value of the %oth kernel argument" % (i,))
                print("Printing kernel output and expected result, set verbose=False to suppress this debug print")
                np.set_printoptions(edgeitems=30)
                print(f"Kernel output ({np.shape(result)}):")
                print(result)
                print(f"Expected ({np.shape(expected)}):")
                print(expected)
                # check if there are NaNs in the output or expected, if so, print where they are
                if lib.isnan(result).any():
                    print("NaNs in kernel output at indices:", lib.where(lib.isnan(result)))
                if lib.isnan(expected).any():
                    print("NaNs in expected result at indices:", lib.where(lib.isnan(expected)))
                # print only the elements that are different
                print("Difference at specific elements:")
                diff = lib.abs(expected - result)
                indices = lib.where(diff > atol)
                print(diff[indices])
            correct = correct and output_test

    if not correct:
        logging.debug("correctness check has found a correctness issue")

    return correct


# these functions facilitate compiling templated kernels with PyCuda
def split_argument_list(argument_list):
    """Split all arguments in a list into types and names."""
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
    """Replace the typename tokens in type_list with their templated typenames."""

    def replace_typename_token(matchobj):
        """Function for a whitespace preserving token regex replace."""
        # replace only the match, leaving the whitespace around it as is
        return matchobj.group(1) + templated_typenames[matchobj.group(2)] + matchobj.group(3)

    for i, arg_type in enumerate(type_list):
        for k, v in templated_typenames.items():
            # if the templated typename occurs as a token in the string, meaning that it is enclosed in
            # beginning of string or whitespace, and end of string, whitespace or star
            regex = r"(^|\s+)(" + k + r")($|\s+|\*)"
            sub = re.sub(regex, replace_typename_token, arg_type, flags=re.S)
            type_list[i] = sub


def get_templated_typenames(template_parameters, template_arguments):
    """Based on the template parameters and arguments, create dict with templated typenames."""
    templated_typenames = {}
    for i, param in enumerate(template_parameters):
        if "typename " in param:
            typename = param[9:]
            templated_typenames[typename] = template_arguments[i]
    return templated_typenames


def wrap_templated_kernel(kernel_string, kernel_name):
    """Rewrite kernel_string to insert wrapper function for templated kernel."""
    # parse kernel_name to find template_arguments and real kernel name
    name = kernel_name.split("<")[0]
    template_arguments = re.search(r".*?<(.*)>", kernel_name, re.S).group(1).split(",")

    # parse templated kernel definition
    # relatively strict regex that does not allow nested template parameters like vector<TF>
    # within the template parameter list
    regex = (
        r"template\s*<([^>]*?)>\s*__global__\s+void\s+(__launch_bounds__\([^\)]+?\)\s+)?" + name + r"\s*\((.*?)\)\s*\{"
    )
    match = re.search(regex, kernel_string, re.S)
    if not match:
        raise ValueError("could not find templated kernel definition")

    template_parameters = match.group(1).split(",")
    argument_list = match.group(3).split(",")
    # remove extra whitespace around 'type name' strings
    argument_list = [s.strip() for s in argument_list]

    type_list, name_list = split_argument_list(argument_list)

    templated_typenames = get_templated_typenames(template_parameters, template_arguments)
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
    template_instantiation = f"template __device__ void {kernel_name}(" + ", ".join(type_list) + ");\n"

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
