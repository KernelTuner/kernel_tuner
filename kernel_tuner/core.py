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

_KernelInstance = namedtuple("_KernelInstance", ["name", "kernel_source", "kernel_string", "temp_files", "threads", "grid", "params", "arguments"])

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

    def __init__(self, kernel_sources, lang):
        if not isinstance(kernel_sources, list):
            kernel_sources = [kernel_sources]
        self.kernel_sources = kernel_sources
        if lang is None:
            if callable(self.kernel_sources[0]):
                raise TypeError("Please specify language when using a code generator function")
            kernel_string = self.get_kernel_string(0)
            lang = util.detect_language(kernel_string)

        # The validity of lang is checked later, when creating the DeviceInterface
        self.lang = lang

    def get_kernel_string(self, index=0, params=None):
        """ retrieve the kernel source with the given index and return as a string

        See util.get_kernel_string() for details.

        :param index: Index of the kernel source in the list of sources.
        :type index: int

        :param params: Dictionary containing the tunable parameters for this specific
            kernel instance, only needed when kernel_source is a generator.
        :type param: dict

        :returns: A string containing the kernel code.
        :rtype: string
        """
        #logging.debug('get_kernel_string called with %s', str(kernel_source))
        logging.debug('get_kernel_string called')

        kernel_source = self.kernel_sources[index]
        return util.get_kernel_string(kernel_source, params)


    def prepare_list_of_files(self, kernel_name, params, grid, threads, block_size_names):
        """ prepare the kernel string along with any additional files

        The first file in the list is allowed to include or read in the others
        The files beyond the first are considered additional files that may also contain tunable parameters

        For each file beyond the first this function creates a temporary file with
        preprocessors statements inserted. Occurences of the original filenames in the
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
                raise ValueError('When passing multiple kernel sources, the secondary entries must be filenames')

            ks = self.get_kernel_string(i, params)
            # add preprocessor statements
            n, ks = util.prepare_kernel_string(kernel_name, ks, params, grid, threads, block_size_names)

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
            # replace occurences of the additional file's name in the first kernel_string with the name of the temp file
            kernel_string = kernel_string.replace(f, temp_file)

        return name, kernel_string, temp_files


    def get_user_suffix(self, index=0):
        """ Get the suffix of the kernel filename, if the user specified one. Return None otherwise.
        """
        if util.looks_like_a_filename(self.kernel_sources[index]) and ("." in self.kernel_sources[index]):
            return "." + self.kernel_sources[index].split(".")[-1]
        return None

    def get_suffix(self, index=0):
        """ Return a suitable suffix for a kernel filename.

        This uses the user-specified suffix if available, or one based on the
        lang/backend otherwise.
        """

        # TODO: Consider delegating this to the backend
        suffix = self.get_user_suffix(index)
        if suffix is not None:
            return suffix

        _suffixes = {'CUDA': '.cu', 'OpenCL': '.cl', 'C': '.c'}
        try:
            return _suffixes[self.lang]
        except KeyError:
            return ".c"

    def check_argument_lists(self, kernel_name, arguments):
        """ Check if the kernel arguments have the correct types

        This is done by calling util.check_argument_list on each kernel string.
        """
        for i, f in enumerate(self.kernel_sources):
            if not callable(f):
                util.check_argument_list(kernel_name, self.get_kernel_string(i), arguments)
            else:
                logging.debug("Checking of arguments list not supported yet for code generators.")


class DeviceInterface(object):
    """Class that offers a High-Level Device Interface to the rest of the Kernel Tuner"""

    def __init__(self, kernel_source, device=0, platform=0, lang=None, quiet=False, compiler=None, compiler_options=None, iterations=7):
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

    def benchmark(self, func, gpu_args, instance, verbose):
        """benchmark the kernel instance"""
        logging.debug('benchmark ' + instance.name)
        logging.debug('thread block dimensions x,y,z=%d,%d,%d', *instance.threads)
        logging.debug('grid dimensions x,y,z=%d,%d,%d', *instance.grid)

        result = None
        try:
            result = self.dev.benchmark(func, gpu_args, instance.threads, instance.grid)
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
        return result

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
            if (verify or answer[i] is not None) and isinstance(arg, numpy.ndarray):
                result_host.append(numpy.zeros_like(arg))
                self.dev.memcpy_dtoh(result_host[-1], gpu_args[i])
            else:
                result_host.append(None)

        #if the user has specified a custom verify function, then call it, else use default based on numpy allclose
        if verify:
            correct = verify(answer, result_host, atol=atol)
        else:
            correct = _default_verify_function(instance, answer, result_host, atol, verbose)

        if not correct:
            raise Exception("Kernel result verification failed for: " + util.get_config_string(instance.params))
        return True

    def compile_and_benchmark(self, kernel_source, gpu_args, params, kernel_options, tuning_options):
        """ Compile and benchmark a kernel instance based on kernel strings and parameters """

        instance_string = util.get_instance_string(params)

        logging.debug('compile_and_benchmark ' + instance_string)
        mem_usage = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0, 1)
        logging.debug('Memory usage : %2.2f MB', mem_usage)

        verbose = tuning_options.verbose

        instance = self.create_kernel_instance(kernel_source, kernel_options, params, verbose)
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
            result = self.benchmark(func, gpu_args, instance, verbose)

        except Exception as e:
            #dump kernel_string to temp file
            temp_filenames = instance.prepare_temp_files_for_error_msg()
            print("Error while compiling or benchmarking, see source files: " + " ".join(temp_filenames))
            raise e

        #clean up any temporary files, if no error occured
        instance.delete_temp_files()

        return result

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

    def create_kernel_instance(self, kernel_source, kernel_options, params, verbose):
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
        name, kernel_string, temp_files = kernel_source.prepare_list_of_files(kernel_options.kernel_name, params, grid, threads, kernel_options.block_size_names)

        #collect everything we know about this instance and return it
        return KernelInstance(name, kernel_source, kernel_string, temp_files, threads, grid, params, kernel_options.arguments)

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
