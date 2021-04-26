""" The simulation runner for sequentially tuning the parameter space based on cached data """
from __future__ import print_function

from collections import OrderedDict
import logging

from kernel_tuner.util import get_config_string, store_cache, process_metrics, print_config_output, get_instance_string


class SimulationLangFunction(object):
    """Compatibility class for supplying simulated device information based on CudaFunctions"""

    def __init__(self, lang, device=0, iterations=7, compiler_options=None):
        self.allocations = []
        self.texrefs = []
        self.use_nvml = False
        self.smem_size = 0
        cc = "00"
        self.cc = str(cc[0]) + str(cc[1])
        self.iterations = iterations
        self.current_module = None
        self.compiler_options = compiler_options or []

        env = dict()
        env["device_name"] = "Simulation"
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        env["device_properties"] = None
        if lang == "CUDA":
            env["cuda_version"] = None
            env["compute_capability"] = self.cc
        elif lang == "OpenCL":
            env["platform_name"] = None
            env["platform_version"] = None
            env["device_version"] = None
            env["opencl_c_version"] = None
            env["driver_version"] = None
        elif lang == "C":
            self.nvcc_available = False
            self.max_threads = None
            self.lib = None
            self.using_openmp = False
            env["CC Version"] = None
        self.env = env
        self.name = env["device_name"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return


class SimulationDeviceInterface(object):
    """Compatibily class for DeviceInterface that offers a High-Level Device Interface to the rest of the Kernel Tuner"""

    def __init__(self, kernel_source, device=0, platform=0, quiet=False, compiler=None, compiler_options=None, iterations=7):
        """ Instantiate the DeviceInterface, based on language in kernel source

        :param kernel_source The kernel sources
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
            Currently supported: "CUDA", "OpenCL", or "C"
        :type lang: string

        :param compiler_options: The compiler options to use when compiling kernels for this device.
        :type compiler_options: list of strings

        :param iterations: Number of iterations to be used when benchmarking using this device.
        :type iterations: int

        :param times: Return the execution time of all iterations.
        :type times: bool

        """
        lang = kernel_source.lang

        logging.debug('DeviceInterface instantiated, lang=%s', lang)

        if lang not in ('CUDA', 'OpenCL', 'C'):
            raise ValueError("Sorry, support for languages other than CUDA, OpenCL, or C is not implemented yet")
        self.lang = lang
        self.dev = SimulationLangFunction(self.lang, device, iterations, compiler_options)
        self.max_threads = 1024
        self.units = None
        self._name = self.dev.name
        self.quiet = quiet
        self.device_access_error = SystemError("Device not accessible in simulation mode")

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        if not self.quiet:
            print("Simulating: " + value)

    def __enter__(self):
        return self

    def benchmark(self, func, gpu_args, instance, verbose):
        """benchmark the kernel instance"""
        logging.debug('benchmark ' + instance.name)
        logging.debug('thread block dimensions x,y,z=%d,%d,%d', *instance.threads)
        logging.debug('grid dimensions x,y,z=%d,%d,%d', *instance.grid)
        raise self.device_access_error

    def check_kernel_output(self, func, gpu_args, instance, answer, atol, verify, verbose):
        """runs the kernel once and checks the result against answer"""
        logging.debug('check_kernel_output')
        raise self.device_access_error

    def compile_and_benchmark(self, kernel_source, gpu_args, params, kernel_options, tuning_options):
        """ Compile and benchmark a kernel instance based on kernel strings and parameters """
        instance_string = get_instance_string(params)
        logging.debug('compile_and_benchmark ' + instance_string)
        raise self.device_access_error

    def compile_kernel(self, instance, verbose):
        """compile the kernel for this specific instance"""
        logging.debug('compile_kernel ' + instance.name)
        raise self.device_access_error

    def copy_constant_memory_args(self, cmem_args):
        """adds constant memory arguments to the most recently compiled module, if using CUDA"""
        raise self.device_access_error

    def copy_texture_memory_args(self, texmem_args):
        """adds texture memory arguments to the most recently compiled module, if using CUDA"""
        raise self.device_access_error

    def create_kernel_instance(self, kernel_source, kernel_options, params, verbose):
        """create kernel instance from kernel source, parameters, problem size, grid divisors, and so on"""
        raise self.device_access_error

    def get_environment(self):
        """Return dictionary with information about the environment"""
        return self.dev.env

    def memcpy_dtoh(self, dest, src):
        """perform a device to host memory copy"""
        raise self.device_access_error

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the kernel, allocates gpu mem if necessary"""
        raise self.device_access_error

    def run_kernel(self, func, gpu_args, instance):
        """ Run a compiled kernel instance on a device """
        logging.debug('run_kernel %s', instance.name)
        logging.debug('thread block dims (%d, %d, %d)', *instance.threads)
        logging.debug('grid dims (%d, %d, %d)', *instance.grid)
        raise self.device_access_error

    def __exit__(self, *exc):
        if hasattr(self, 'dev'):
            self.dev.__exit__(*exc)


class SimulationRunner(object):
    """ SimulationRunner is used for tuning with a single process/thread """

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers):
        """ Instantiate the SimulationRunner

        :param kernel_source: The kernel source
        :type kernel_source: kernel_tuner.core.KernelSource

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param device_options: A dictionary with all options for the device
            on which the kernel should be tuned.
        :type device_options: kernel_tuner.interface.Options

        :param iterations: The number of iterations used for benchmarking
            each kernel instance.
        :type iterations: int
        """

        # #detect language and create high-level device interface
        self.dev = SimulationDeviceInterface(kernel_source, iterations=iterations, **device_options).__enter__()

        self.quiet = device_options.quiet
        self.kernel_source = kernel_source

        self.simulation_mode = True

    def __enter__(self):
        return self

    def run(self, parameter_space, kernel_options, tuning_options):
        """ Iterate through the entire parameter space using a single Python process

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param tuning_options: A dictionary with all options regarding the tuning
            process.
        :type tuning_options: kernel_tuner.iterface.Options

        :returns: A list of dictionaries for executed kernel configurations and their
            execution times. And a dictionary that contains information
            about the hardware/software environment on which the tuning took place.
        :rtype: list(dict()), dict()

        """
        logging.debug('simulation runner started for ' + kernel_options.kernel_name)

        results = []

        # iterate over parameter space
        for element in parameter_space:

            # check if element is in the cache
            x_int = ",".join([str(i) for i in element])
            if tuning_options.cache and x_int in tuning_options.cache:
                results.append(tuning_options.cache[x_int])
                continue

            # if the element is not in the cache, raise an error
            logging.debug('parameter element not in cache')
            print(element)
            raise ValueError("Parameter element not in cache - in simulation mode, all parameter elements must be present in the cache")

        return results, self.dev.get_environment()

    def __exit__(self, *exc):
        if hasattr(self, 'dev'):
            self.dev.__exit__(*exc)
