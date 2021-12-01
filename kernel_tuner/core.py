""" Module for grouping the core functionality needed by most runners """
from __future__ import print_function

import re
from collections import namedtuple
import resource
import logging
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np

from kernel_tuner.cupy import CupyFunctions
from kernel_tuner.cuda import CudaFunctions
from kernel_tuner.opencl import OpenCLFunctions
from kernel_tuner.c import CFunctions
from kernel_tuner.nvml import NVMLObserver
import kernel_tuner.util as util


try:
    import torch
except ImportError:
    torch = util.TorchPlaceHolder()



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

    def __init__(self, kernel_name, kernel_sources, lang):
        if not isinstance(kernel_sources, list):
            kernel_sources = [kernel_sources]
        self.kernel_sources = kernel_sources
        self.kernel_name = kernel_name
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
            n, ks = util.prepare_kernel_string(kernel_name, ks, params, grid, threads, block_size_names, self.lang)

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

        _suffixes = {
            'CUDA': '.cu',
            'OpenCL': '.cl',
            'C': '.c'
        }
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

    def __init__(self, kernel_source, device=0, platform=0, quiet=False, compiler=None, compiler_options=None, iterations=7, observers=None):
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
            dev = CudaFunctions(device, compiler_options=compiler_options, iterations=iterations, observers=observers)
        elif lang.upper() == "CUPY":
            dev = CupyFunctions(device, compiler_options=compiler_options, iterations=iterations, observers=observers)
        elif lang == "OpenCL":
            dev = OpenCLFunctions(device, platform, compiler_options=compiler_options, iterations=iterations, observers=observers)
        elif lang == "C":
            dev = CFunctions(compiler=compiler, compiler_options=compiler_options, iterations=iterations)
        else:
            raise ValueError("Sorry, support for languages other than CUDA, OpenCL, or C is not implemented yet")

        #look for NVMLObserver in observers, if present, enable special tunable parameters through nvml
        self.use_nvml = False
        if observers:
            for obs in observers:
                if isinstance(obs, NVMLObserver):
                    self.nvml = obs.nvml
                    self.use_nvml = True

        self.lang = lang
        self.dev = dev
        self.units = dev.units
        self.name = dev.name
        self.max_threads = dev.max_threads
        if not quiet:
            print("Using: " + self.dev.name)

        dev.__enter__()

    def __enter__(self):
        return self

    def benchmark(self, func, gpu_args, instance, verbose):
        """benchmark the kernel instance"""
        logging.debug('benchmark ' + instance.name)
        logging.debug('thread block dimensions x,y,z=%d,%d,%d', *instance.threads)
        logging.debug('grid dimensions x,y,z=%d,%d,%d', *instance.grid)

        if self.use_nvml:
            if "nvml_pwr_limit" in instance.params:
                new_limit = int(instance.params["nvml_pwr_limit"] * 1000)    #user specifies in Watt, but nvml uses milliWatt
                if self.nvml.pwr_limit != new_limit:
                    self.nvml.pwr_limit = new_limit
            if "nvml_gr_clock" in instance.params:
                self.nvml.gr_clock = instance.params["nvml_gr_clock"]
            if "nvml_sm_clock" in instance.params:
                self.nvml.sm_clock = instance.params["nvml_sm_clock"]
            if "nvml_mem_clock" in instance.params:
                self.nvml.mem_clock = instance.params["nvml_mem_clock"]

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
                    print(f"skipping config {util.get_instance_string(instance.params)} reason: too many resources requested for launch")
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
            if (verify or answer[i] is not None) and isinstance(arg, (np.ndarray, cp.ndarray, torch.Tensor)):
                self.dev.memcpy_htod(gpu_args[i], arg)

        #run the kernel
        check = self.run_kernel(func, gpu_args, instance)
        if not check:
            return True    #runtime failure occured that should be ignored, skip correctness check

        #retrieve gpu results to host memory
        result_host = []
        for i, arg in enumerate(instance.arguments):
            if (verify or answer[i] is not None) and isinstance(arg, (np.ndarray, cp.ndarray)):
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
                result_host.append(None)

        #if the user has specified a custom verify function, then call it, else use default based on numpy allclose
        if verify:
            correct = verify(answer, result_host, atol=atol)
        else:
            correct = _default_verify_function(instance, answer, result_host, atol, verbose)

        if not correct:
            raise RuntimeError("Kernel result verification failed for: " + util.get_config_string(instance.params))
        return True

    def compile_and_benchmark(self, kernel_source, gpu_args, params, kernel_options, tuning_options):
        """ Compile and benchmark a kernel instance based on kernel strings and parameters """

        instance_string = util.get_instance_string(params)

        logging.debug('compile_and_benchmark ' + instance_string)
        mem_usage = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
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

            #add shared memory arguments to compiled module
            if kernel_options.smem_args is not None:
                self.dev.copy_shared_memory_args(util.get_smem_args(kernel_options.smem_args, params))
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
            func = self.dev.compile(instance)
        except Exception as e:
            #compiles may fail because certain kernel configurations use too
            #much shared memory for example, the desired behavior is to simply
            #skip over this configuration and try the next one
            shared_mem_error_messages = ["uses too much shared data", "local memory limit exceeded"]
            if any(msg in str(e) for msg in shared_mem_error_messages):
                logging.debug('compile_kernel failed due to kernel using too much shared memory')
                if verbose:
                    print(f"skipping config {util.get_instance_string(instance.params)} reason: too much shared memory used")
            else:
                logging.debug('compile_kernel failed due to error: ' + str(e))
                print("Error while compiling:", instance.name)
                raise e
        return func

    def copy_shared_memory_args(self, smem_args):
        """adds shared memory arguments to the most recently compiled module, if using CUDA"""
        if self.lang == "CUDA":
            self.dev.copy_shared_memory_args(smem_args)
        else:
            raise RuntimeError("Error cannot copy shared memory arguments when language is not CUDA")

    def copy_constant_memory_args(self, cmem_args):
        """adds constant memory arguments to the most recently compiled module, if using CUDA"""
        if self.lang == "CUDA":
            self.dev.copy_constant_memory_args(cmem_args)
        else:
            raise RuntimeError("Error cannot copy constant memory arguments when language is not CUDA")

    def copy_texture_memory_args(self, texmem_args):
        """adds texture memory arguments to the most recently compiled module, if using CUDA"""
        if self.lang == "CUDA":
            self.dev.copy_texture_memory_args(texmem_args)
        else:
            raise RuntimeError("Error cannot copy texture memory arguments when language is not CUDA")

    def create_kernel_instance(self, kernel_source, kernel_options, params, verbose):
        """create kernel instance from kernel source, parameters, problem size, grid divisors, and so on"""
        grid_div = (kernel_options.grid_div_x, kernel_options.grid_div_y, kernel_options.grid_div_z)

        #insert default block_size_names if needed
        if not kernel_options.block_size_names:
            kernel_options.block_size_names = util.default_block_size_names

        #setup thread block and grid dimensions
        threads, grid = util.setup_block_and_grid(kernel_options.problem_size, grid_div, params, kernel_options.block_size_names)
        if np.prod(threads) > self.dev.max_threads:
            if verbose:
                print(f"skipping config {util.get_instance_string(params)} reason: too many threads per block")
            return None

        #obtain the kernel_string and prepare additional files, if any
        name, kernel_string, temp_files = kernel_source.prepare_list_of_files(kernel_options.kernel_name, params, grid, threads,
                                                                              kernel_options.block_size_names)

        #check for templated kernel
        if kernel_source.lang == "CUDA" and "<" in name and ">" in name:
            kernel_string, name = wrap_templated_kernel(kernel_string, name)

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

    def __exit__(self, *exc):
        if hasattr(self, 'dev'):
            self.dev.__exit__(*exc)


def _default_verify_function(instance, answer, result_host, atol, verbose):
    """default verify function based on np.allclose"""

    #first check if the length is the same
    if len(instance.arguments) != len(answer):
        raise TypeError("The length of argument list and provided results do not match.")
    #for each element in the argument list, check if the types match
    for i, arg in enumerate(instance.arguments):
        if answer[i] is not None:    #skip None elements in the answer list
            if isinstance(answer[i], (np.ndarray, cp.ndarray)) and isinstance(arg, (np.ndarray, cp.ndarray)):
                if answer[i].dtype != arg.dtype:
                    raise TypeError(f"Element {i} of the expected results list is not of the same dtype as the kernel output: " +
                                    str(answer[i].dtype) + " != " + str(arg.dtype) + ".")
                if answer[i].size != arg.size:
                    raise TypeError(f"Element {i} of the expected results list has a size different from " + "the kernel argument: " +
                                    str(answer[i].size) + " != " + str(arg.size) + ".")
            elif isinstance(answer[i], torch.Tensor) and isinstance(arg, torch.Tensor):
                if answer[i].dtype != arg.dtype:
                    raise TypeError(f"Element {i} of the expected results list is not of the same dtype as the kernel output: " +
                                    str(answer[i].dtype) + " != " + str(arg.dtype) + ".")
                if answer[i].size() != arg.size():
                    raise TypeError(f"Element {i} of the expected results list has a size different from " + "the kernel argument: " +
                                    str(answer[i].size) + " != " + str(arg.size) + ".")

            elif isinstance(answer[i], np.number) and isinstance(arg, np.number):
                if answer[i].dtype != arg.dtype:
                    raise TypeError(f"Element {i} of the expected results list is not the same as the kernel output: " + str(answer[i].dtype) +
                                    " != " + str(arg.dtype) + ".")
            else:
                #either answer[i] and argument have different types or answer[i] is not a numpy type
                if not isinstance(answer[i], (np.ndarray, cp.ndarray, torch.Tensor)) or not isinstance(answer[i], np.number):
                    raise TypeError(f"Element {i} of expected results list is not a numpy/cupy ndarray, torch Tensor or numpy scalar.")
                else:
                    raise TypeError(f"Element {i} of expected results list and kernel arguments have different types.")

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
            if any([isinstance(array, cp.ndarray) for array in [expected, result]]):
                output_test = cp.allclose(expected, result, atol=atol)
            elif isinstance(expected, torch.Tensor) and isinstance(result, torch.Tensor):
                output_test = torch.allclose(expected, result, atol=atol)
            else:
                output_test = np.allclose(expected, result, atol=atol)

            if not output_test and verbose:
                print("Error: " + util.get_config_string(instance.params) + " detected during correctness check")
                print("this error occured when checking value of the %oth kernel argument" % (i, ))
                print("Printing kernel output and expected result, set verbose=False to suppress this debug print")
                np.set_printoptions(edgeitems=50)
                print("Kernel output:")
                print(result)
                print("Expected:")
                print(expected)
            correct = correct and output_test

    if not correct:
        logging.debug('correctness check has found a correctness issue')

    return correct



#these functions facilitate compiling templated kernels with PyCuda
def split_argument_list(argument_list):
    """split all arguments in a list into types and names"""
    regex = r"(.*[\s*]+)(.*)?"
    type_list = []
    name_list = []
    for arg in argument_list:
        match = re.match(regex, arg, re.S)
        if not match:
            raise ValueError("error parsing templated kernel argument list")
        type_list.append(re.sub(r"\s+", " ", match.group(1).strip(), re.S))
        name_list.append(match.group(2).strip())
    return type_list, name_list

def apply_template_typenames(type_list, templated_typenames):
    """replace the typename tokens in type_list with their templated typenames"""
    def replace_typename_token(matchobj):
        """function for a whitespace preserving token regex replace"""
        #replace only the match, leaving the whitespace around it as is
        return matchobj.group(1) + templated_typenames[matchobj.group(2)] + matchobj.group(3)
    for i, arg_type in enumerate(type_list):
        for k,v in templated_typenames.items():
            #if the templated typename occurs as a token in the string, meaning that it is enclosed in
            #beginning of string or whitespace, and end of string, whitespace or star
            regex = r"(^|\s+)(" + k + r")($|\s+|\*)"
            sub = re.sub(regex, replace_typename_token, arg_type, re.S)
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
    #parse kernel_name to find template_arguments and real kernel name
    name = kernel_name.split("<")[0]
    template_arguments = re.search(r".*?<(.*)>", kernel_name, re.S).group(1).split(',')

    #parse templated kernel definition
    #relatively strict regex that does not allow nested template parameters like vector<TF>
    #within the template parameter list
    regex = r"template\s*<([^>]*?)>\s*__global__\s+void\s+" + name + r"\s*\((.*?)\)\s*\{"
    match = re.search(regex, kernel_string, re.S)
    if not match:
        raise ValueError("could not find templated kernel definition")

    template_parameters = match.group(1).split(',')
    argument_list = match.group(2).split(',')
    argument_list = [s.strip() for s in argument_list] #remove extra whitespace around 'type name' strings

    type_list, name_list = split_argument_list(argument_list)

    templated_typenames = get_templated_typenames(template_parameters, template_arguments)
    apply_template_typenames(type_list, templated_typenames)

    #replace __global__ with __device__ in the templated kernel definition
    #could do a more precise replace, but __global__ cannot be used elsewhere in the definition
    definition = match.group(0).replace("__global__", "__device__")

    #generate code for the compile-time template instantiation
    template_instantiation = f"template __device__ void {kernel_name}(" + ", ".join(type_list) + ");\n"

    #generate code for the wrapper kernel
    new_arg_list = ", ".join([" ".join((a, b)) for a, b in zip(type_list, name_list)])
    wrapper_function = "\nextern \"C\" __global__ void " + name + "_wrapper(" + new_arg_list + ") {\n  " + \
       kernel_name + "(" + ", ".join(name_list) + ");\n}\n"

    #copy kernel_string, replace definition and append template instantiation and wrapper function
    new_kernel_string = kernel_string[:]
    new_kernel_string = new_kernel_string.replace(match.group(0), definition)
    new_kernel_string += "\n" + template_instantiation
    new_kernel_string += wrapper_function

    return new_kernel_string, name + "_wrapper"
