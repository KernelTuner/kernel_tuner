""" Module for grouping the core functionality needed by most runners """
from __future__ import print_function

from collections import namedtuple
import resource
import logging
import numpy

from kernel_tuner.cuda import CudaFunctions
from kernel_tuner.opencl import OpenCLFunctions
from kernel_tuner.c import CFunctions
import kernel_tuner.util as util

KernelInstance = namedtuple("KernelInstance", ["name", "kernel_string", "temp_files", "threads", "grid", "params", "arguments"])

class DeviceInterface(object):
    """Class that offers a High-Level Device Interface to the rest of the Kernel Tuner"""

    def __init__(self, original_kernel, device=0, platform=0, lang=None, quiet=False, compiler=None, compiler_options=None, iterations=7):
        """ Instantiate the DeviceInterface, based on language in kernel source

        :param original_kernel: The source of the kernel as passed to tune_kernel
        :type original_kernel: kernel source as a string or a list of strings denoting filenames

        :param device: CUDA/OpenCL device to use, in case you have multiple
            CUDA-capable GPUs or OpenCL devices you may use this to select one,
            0 by default. Ignored if you are tuning host code by passing lang="C".
        :type device: int

        :param platform: OpenCL platform to use, in case you have multiple
            OpenCL platforms you may use this to select one,
            0 by default. Ignored if not using OpenCL.
        :type device: int

        :param lang: Specifies the language used for GPU kernels. The kernel_tuner
            automatically detects the language, but if it fails, you may specify
            the language using this argument, currently supported: "CUDA", "OpenCL", or "C"
        :type lang: string

        :param compiler_options: The compiler options to use when compiling kernels for this device.
        :type compiler_options: list of strings

        :param iterations: Number of iterations to be used when benchmarking using this device.
        :type iterations: int

        :param times: Return the execution time of all iterations.
        :type times: bool

        """
        logging.debug('DeviceInterface instantiated, lang=%s', lang)

        lang = util.detect_language(lang, original_kernel)
        if lang == "CUDA":
            dev = CudaFunctions(device, compiler_options=compiler_options, iterations=iterations)
        elif lang == "OpenCL":
            dev = OpenCLFunctions(device, platform, compiler_options=compiler_options, iterations=iterations)
        elif lang == "C":
            dev = CFunctions(compiler=compiler, compiler_options=compiler_options, iterations=iterations)
        else:
            raise Exception("Sorry, support for languages other than CUDA, OpenCL, or C is not implemented yet")
        self.lang = lang
        self.dev = dev
        self.units = dev.units
        self.name = dev.name
        if not quiet:
            print("Using: " + self.dev.name)

    def benchmark(self, func, gpu_args, instance, times, verbose):
        """benchmark the kernel instance"""
        logging.debug('benchmark ' + instance.name)
        logging.debug('thread block dimensions x,y,z=%d,%d,%d', *instance.threads)
        logging.debug('grid dimensions x,y,z=%d,%d,%d', *instance.grid)

        time = None
        try:
            time = self.dev.benchmark(func, gpu_args, instance.threads, instance.grid, times)
        except Exception as e:
            #some launches may fail because too many registers are required
            #to run the kernel given the current thread block size
            #the desired behavior is to simply skip over this configuration
            #and proceed to try the next one
            skippable_exceptions = ["too many resources requested for launch", "OUT_OF_RESOURCES", "INVALID_WORK_GROUP_SIZE"]
            if any([skip_str in str(e) for skip_str in skippable_exceptions]):
                logging.debug('benchmark fails due to runtime failure too many resources required')
                if verbose:
                    print("skipping config", instance.name, "reason: too many resources requested for launch")
            else:
                logging.debug('benchmark encountered runtime failure: ' + str(e))
                print("Error while benchmarking:", instance.name)
                raise e
        return time

    def check_kernel_output(self, func, gpu_args, instance, answer, atol, verify, verbose):
        """runs the kernel once and checks the result against answer"""
        logging.debug('check_kernel_output')

        #if not using custom verify function, check if the length is the same
        if not verify and len(instance.arguments) != len(answer):
            raise TypeError("The length of argument list and provided results do not match.")

        #re-copy original contents of output arguments to GPU memory, to overwrite any changes
        #by earlier kernel runs
        for i, arg in enumerate(instance.arguments):
            if verify or answer[i] is not None:
                if isinstance(arg, numpy.ndarray):
                    self.dev.memcpy_htod(gpu_args[i], arg)

        #run the kernel
        check = self.run_kernel(func, gpu_args, instance)
        if not check:
            return True #runtime failure occured that should be ignored, skip correctness check

        #retrieve gpu results to host memory
        result_host = []
        for i, arg in enumerate(instance.arguments):
            if verify or answer[i] is not None:
                if isinstance(arg, numpy.ndarray):
                    result_host.append(numpy.zeros_like(arg))
                    self.dev.memcpy_dtoh(result_host[-1], gpu_args[i])
            else:
                result_host.append(None)

        #if the user has specified a custom verify function, then call it, else use default based on numpy allclose
        if verify:
            try:
                correct = verify(answer, result_host, atol=atol)
            except TypeError:
                correct = verify(answer, result_host)
        else:
            correct = _default_verify_function(instance, answer, result_host, atol, verbose)

        if not correct:
            raise Exception("Kernel result verification failed for: " + util.get_config_string(instance.params))
        return True

    def compile_and_benchmark(self, gpu_args, params, kernel_options, tuning_options):
        """ Compile and benchmark a kernel instance based on kernel strings and parameters """

        instance_string = util.get_instance_string(params)

        logging.debug('compile_and_benchmark ' + instance_string)
        mem_usage = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1)
        logging.debug('Memory usage : %2.2f MB', mem_usage)

        verbose = tuning_options.verbose

        instance = self.create_kernel_instance(kernel_options, params, verbose)
        if instance is None:
            return None

        try:
            #compile the kernel
            func = self.compile_kernel(instance, verbose)
            if func is None:
                return None

            #add constant memory arguments to compiled module
            if kernel_options.cmem_args is not None:
                self.dev.copy_constant_memory_args(kernel_options.cmem_args)
            #add texture memory arguments to compiled module
            if kernel_options.texmem_args is not None:
                self.dev.copy_texture_memory_args(kernel_options.texmem_args)

            #test kernel for correctness and benchmark
            if tuning_options.answer is not None or tuning_options.verify is not None:
                self.check_kernel_output(func, gpu_args, instance, tuning_options.answer, tuning_options.atol, tuning_options.verify, verbose)

            #benchmark
            time = self.benchmark(func, gpu_args, instance, tuning_options.times, verbose)

        except Exception as e:
            #dump kernel_string to temp file
            temp_filename = util.get_temp_filename(suffix=".c")
            util.write_file(temp_filename, instance.kernel_string)
            print("Error while compiling or benchmarking, see source files: " + temp_filename + " ".join(instance.temp_files.values()))
            raise e

        #clean up any temporary files, if no error occured
        for v in instance.temp_files.values():
            util.delete_temp_file(v)

        return time

    def compile_kernel(self, instance, verbose):
        """compile the kernel for this specific instance"""
        logging.debug('compile_kernel ' + instance.name)

        #compile kernel_string into device func
        func = None
        try:
            func = self.dev.compile(instance.name, instance.kernel_string)
        except Exception as e:
            #compiles may fail because certain kernel configurations use too
            #much shared memory for example, the desired behavior is to simply
            #skip over this configuration and try the next one
            if "uses too much shared data" in str(e):
                logging.debug('compile_kernel failed due to kernel using too much shared memory')
                if verbose:
                    print("skipping config", instance.name, "reason: too much shared memory used")
            else:
                logging.debug('compile_kernel failed due to error: ' + str(e))
                print("Error while compiling:", instance.name)
                raise e
        return func

    def copy_constant_memory_args(self, cmem_args):
        """adds constant memory arguments to the most recently compiled module, if using CUDA"""
        if self.lang == "CUDA":
            self.dev.copy_constant_memory_args(cmem_args)
        else:
            raise Exception("Error cannot copy constant memory arguments when language is not CUDA")

    def copy_texture_memory_args(self, texmem_args):
        """adds texture memory arguments to the most recently compiled module, if using CUDA"""
        if self.lang == "CUDA":
            self.dev.copy_texture_memory_args(texmem_args)
        else:
            raise Exception("Error cannot copy texture memory arguments when language is not CUDA")

    def create_kernel_instance(self, kernel_options, params, verbose):
        """create kernel instance from kernel source, parameters, problem size, grid divisors, and so on"""
        instance_string = util.get_instance_string(params)
        grid_div = (kernel_options.grid_div_x, kernel_options.grid_div_y, kernel_options.grid_div_z)

        #insert default block_size_names if needed
        if not kernel_options.block_size_names:
            kernel_options.block_size_names = util.default_block_size_names

        #setup thread block and grid dimensions
        threads, grid = util.setup_block_and_grid(kernel_options.problem_size, grid_div, params, kernel_options.block_size_names)
        if numpy.prod(threads) > self.dev.max_threads:
            if verbose:
                print("skipping config", instance_string, "reason: too many threads per block")
            return None

        #obtain the kernel_string and prepare additional files, if any
        temp_files = dict()
        kernel_source = kernel_options.kernel_string
        if not isinstance(kernel_source, list):
            kernel_source = [kernel_source]
        name, kernel_string, temp_files = util.prepare_list_of_files(kernel_options.kernel_name, kernel_source, params, grid, threads, kernel_options.block_size_names)

        #collect everything we know about this instance and return it
        return KernelInstance(name, kernel_string, temp_files, threads, grid, params, kernel_options.arguments)

    def get_environment(self):
        """Return dictionary with information about the environment"""
        return self.dev.env

    def memcpy_dtoh(self, dest, src):
        """perform a device to host memory copy"""
        self.dev.memcpy_dtoh(dest, src)

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the kernel, allocates gpu mem if necessary"""
        return self.dev.ready_argument_list(arguments)

    def run_kernel(self, func, gpu_args, instance):
        """ Run a compiled kernel instance on a device """
        logging.debug('run_kernel %s', instance.name)
        logging.debug('thread block dims (%d, %d, %d)', *instance.threads)
        logging.debug('grid dims (%d, %d, %d)', *instance.grid)

        try:
            self.dev.run_kernel(func, gpu_args, instance.threads, instance.grid)
        except Exception as e:
            if "too many resources requested for launch" in str(e) or "OUT_OF_RESOURCES" in str(e):
                logging.debug('ignoring runtime failure due to too many resources required')
                return False
            else:
                logging.debug('encountered unexpected runtime failure: ' + str(e))
                raise e
        return True


    def __del__(self):
        if hasattr(self, 'dev'):
            del self.dev



def _default_verify_function(instance, answer, result_host, atol, verbose):
    """default verify function based on numpy.allclose"""

    #first check if the length is the same
    if len(instance.arguments) != len(answer):
        raise TypeError("The length of argument list and provided results do not match.")
    #for each element in the argument list, check if the types match
    for i, arg in enumerate(instance.arguments):
        if answer[i] is not None: #skip None elements in the answer list
            if isinstance(answer[i], numpy.ndarray) and isinstance(arg, numpy.ndarray):
                if answer[i].dtype != arg.dtype:
                    raise TypeError("Element " + str(i)
                                    + " of the expected results list is not of the same dtype as the kernel output: "
                                    + str(answer[i].dtype) + " != " + str(arg.dtype) + ".")
                if answer[i].size != arg.size:
                    raise TypeError("Element " + str(i)
                                    + " of the expected results list has a size different from "
                                    + "the kernel argument: "
                                    + str(answer[i].size) + " != " + str(arg.size) + ".")
            elif isinstance(answer[i], numpy.number) and isinstance(arg, numpy.number):
                if answer[i].dtype != arg.dtype:
                    raise TypeError("Element " + str(i)
                                    + " of the expected results list is not the same as the kernel output: "
                                    + str(answer[i].dtype) + " != " + str(arg.dtype) + ".")
            else:
                #either answer[i] and argument have different types or answer[i] is not a numpy type
                if not isinstance(answer[i], numpy.ndarray) or not isinstance(answer[i], numpy.number):
                    raise TypeError("Element " + str(i)
                                    + " of expected results list is not a numpy array or numpy scalar.")
                else:
                    raise TypeError("Element " + str(i)
                                    + " of expected results list and kernel arguments have different types.")

    def _ravel(a):
        if hasattr(a, 'ravel') and len(a.shape) > 1:
            return a.ravel()
        return a

    def _flatten(a):
        if hasattr(a, 'flatten'):
            return a.flatten()
        return a

    correct = True
    for i, arg in enumerate(instance.arguments):
        expected = answer[i]
        if expected is not None:

            result = _ravel(result_host[i])
            expected = _flatten(expected)
            output_test = numpy.allclose(expected, result, atol=atol)

            if not output_test and verbose:
                print("Error: " + util.get_config_string(instance.params) + " detected during correctness check")
                print("this error occured when checking value of the %oth kernel argument" % (i,))
                print("Printing kernel output and expected result, set verbose=False to suppress this debug print")
                numpy.set_printoptions(edgeitems=50)
                print("Kernel output:")
                print(result)
                print("Expected:")
                print(expected)
            correct = correct and output_test

    if not correct:
        logging.debug('correctness check has found a correctness issue')

    return correct


