"""This module contains all Cupy specific kernel_tuner functions"""
from __future__ import print_function

import re

import logging
import time
import numpy as np

from kernel_tuner.observers import BenchmarkObserver
from kernel_tuner.nvml import nvml

#embedded in try block to be able to generate documentation
#and run tests without cupy installed
try:
    import cupy as cp
except ImportError:
    cp = None


class CupyRuntimeObserver(BenchmarkObserver):
    """ Observer that measures time using CUDA events during benchmarking """
    def __init__(self, dev):
        self.dev = dev
        self.stream = dev.stream
        self.start = dev.start
        self.end = dev.end
        self.times = []

    def after_finish(self):
        self.times.append(cp.cuda.get_elapsed_time(self.start, self.end)) #ms

    def get_results(self):
        results = {"time": np.average(self.times), "times": self.times.copy()}
        self.times = []
        return results


class CupyFunctions(object):
    """Class that groups the Cupy functions on maintains state about the device"""

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """instantiate CudaFunctions object used for interacting with the CUDA device

        Instantiating this object will inspect and store certain device properties at
        runtime, which are used during compilation and/or execution of kernels by the
        kernel tuner. It also maintains a reference to the most recently compiled
        source module for copying data to constant memory before kernel launch.

        :param device: Number of CUDA device to use for this context
        :type device: int

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int
        """
        self.allocations = []
        self.texrefs = []
        if not cp:
            raise ImportError("Error: cupy not installed, please install e.g. using 'pip install cupy'.")

        #select device
        self.dev = dev = cp.cuda.Device(device)
        dev.use()

        #inspect device properties
        self.devprops = dev.attributes
        self.cc = dev.compute_capability
        self.max_threads = self.devprops['MaxThreadsPerBlock']

        self.iterations = iterations
        self.current_module = None
        self.compiler_options = compiler_options or []

        #create a stream and events
        self.stream = cp.cuda.Stream()
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()

        #default dynamically allocated shared memory size, can be overwritten using smem_args
        self.smem_size = 0

        #setup observers
        self.observers = observers or []
        self.observers.append(CupyRuntimeObserver(self))
        for obs in self.observers:
            obs.register_device(self)

        #collect environment information
        env = dict()
        cupy_info = str(cp._cupyx.get_runtime_info()).split("\n")[:-1]
        info_dict = {s.split(":")[0].strip():s.split(":")[1].strip() for s in cupy_info}
        env["device_name"] = info_dict[f'Device {device} Name']

        env["cuda_version"] = cp.cuda.runtime.driverGetVersion()
        env["compute_capability"] = self.cc
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        env["device_properties"] = self.devprops
        self.env = env
        self.name = env["device_name"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the kernel, allocates gpu mem

        :param arguments: List of arguments to be passed to the kernel.
            The order should match the argument list on the CUDA kernel.
            Allowed values are numpy.ndarray, and/or numpy.int32, numpy.float32, and so on.
        :type arguments: list(numpy objects)

        :returns: A list of arguments that can be passed to an CUDA kernel.
        :rtype: list( cupy.ndarray, numpy.int32, ... )
        """
        gpu_args = []
        for arg in arguments:
            # if arg i is a numpy array copy to device
            if isinstance(arg, np.ndarray):
                alloc = cp.array(arg)
                self.allocations.append(alloc)
                gpu_args.append(alloc)
            else: # if not an array, just pass argument along
                gpu_args.append(arg)
        return gpu_args


    def compile(self, kernel_instance):
        """call the CUDA compiler to compile the kernel, return the device function

        :param kernel_name: The name of the kernel to be compiled, used to lookup the
            function after compilation.
        :type kernel_name: string

        :param kernel_string: The CUDA kernel code that contains the function `kernel_name`
        :type kernel_string: string

        :returns: An CUDA kernel that can be called directly.
        :rtype: cupy.RawKernel
        """
        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name

        #try:
        if True:
            #no_extern_c = 'extern "C"' in kernel_string

            compiler_options = ['--std=c++11']  #['-Xcompiler=-Wall']
            if self.compiler_options:
                compiler_options += self.compiler_options

            #self.current_module = self.source_mod(kernel_string, options=compiler_options + ["-e", kernel_name],
            #                                 arch=('compute_' + self.cc) if self.cc != "00" else None,
            #                                 code=('sm_' + self.cc) if self.cc != "00" else None,
            #                                 cache_dir=False, no_extern_c=no_extern_c)

            options = tuple(compiler_options)

            self.current_module = cp.RawModule(code=kernel_string, options=options,
                                               name_expressions=[kernel_name])

            func = self.current_module.get_function(kernel_name)
            self.func = func
            return func

        #except drv.CompileError as e:
        #    if "uses too much shared data" in e.stderr:
        #        raise Exception("uses too much shared data")
        #    else:
        #        raise e
        


    def benchmark(self, func, gpu_args, threads, grid):
        """runs the kernel and measures time repeatedly, returns average time

        Runs the kernel and measures kernel execution time repeatedly, number of
        iterations is set during the creation of CudaFunctions. Benchmark returns
        a robust average, from all measurements the fastest and slowest runs are
        discarded and the rest is included in the returned average. The reason for
        this is to be robust against initialization artifacts and other exceptional
        cases.

        :param func: A cupy kernel compiled for this specific kernel configuration
        :type func: cupy.RawKernel

        :param gpu_args: A list of arguments to the kernel, order should match the
            order in the code. Allowed values are either variables in global memory
            or single values passed by value.
        :type gpu_args: list( cupy.ndarray, numpy.int32, ...)

        :param threads: A tuple listing the number of threads in each dimension of
            the thread block
        :type threads: tuple(int, int, int)

        :param grid: A tuple listing the number of thread blocks in each dimension
            of the grid
        :type grid: tuple(int, int)

        :returns: A dictionary with benchmark results.
        :rtype: dict()
        """
        result = dict()
        self.dev.synchronize()
        for _ in range(self.iterations):
            self.start.record(stream=self.stream)
            for obs in self.observers:
                obs.before_start()
            self.run_kernel(func, gpu_args, threads, grid, stream=self.stream)
            self.end.record(stream=self.stream)
            for obs in self.observers:
                obs.after_start()
            while not self.end.done:
                for obs in self.observers:
                    obs.during()
            for obs in self.observers:
                obs.after_finish()

        for obs in self.observers:
            result.update(obs.get_results())

        return result

    def copy_constant_memory_args(self, cmem_args):
        """adds constant memory arguments to the most recently compiled module

        :param cmem_args: A dictionary containing the data to be passed to the
            device constant memory. The format to be used is as follows: A
            string key is used to name the constant memory symbol to which the
            value needs to be copied. Similar to regular arguments, these need
            to be numpy objects, such as numpy.ndarray or numpy.int32, and so on.
        :type cmem_args: dict( string: numpy.ndarray, ... )
        """
        logging.debug('copy_constant_memory_args called')
        logging.debug('current module: ' + str(self.current_module))

        """
        for k, v in cmem_args.items():
            symbol = self.current_module.get_global(k)[0]
            logging.debug('copying to symbol: ' + str(symbol))
            logging.debug('array to be copied: ')
            logging.debug(v.nbytes)
            logging.debug(v.dtype)
            logging.debug(v.flags)
            drv.memcpy_htod(symbol, v)
        """

    def copy_shared_memory_args(self, smem_args):
        """add shared memory arguments to the kernel"""
        self.smem_size = smem_args["size"]

    def copy_texture_memory_args(self, texmem_args):
        """adds texture memory arguments to the most recently compiled module

        :param texmem_args: A dictionary containing the data to be passed to the
            device texture memory. See tune_kernel().
        :type texmem_args: dict
        """
        filter_mode_map = { 'point': drv.filter_mode.POINT,
                            'linear': drv.filter_mode.LINEAR }
        address_mode_map = { 'border': drv.address_mode.BORDER,
                             'clamp': drv.address_mode.CLAMP,
                             'mirror': drv.address_mode.MIRROR,
                             'wrap': drv.address_mode.WRAP }

        logging.debug('copy_texture_memory_args called')
        logging.debug('current module: ' + str(self.current_module))

        """
        self.texrefs = []
        for k, v in texmem_args.items():
            tex = self.current_module.get_texref(k)
            self.texrefs.append(tex)

            logging.debug('copying to texture: ' + str(k))
            if not isinstance(v, dict):
                data = v
            else:
                data = v['array']
            logging.debug('texture to be copied: ')
            logging.debug(data.nbytes)
            logging.debug(data.dtype)
            logging.debug(data.flags)

            drv.matrix_to_texref(data, tex, order="C")

            if isinstance(v, dict):
                if 'address_mode' in v and v['address_mode'] is not None:
                    # address_mode is set per axis
                    amode = v['address_mode']
                    if not isinstance(amode, list):
                        amode = [ amode ] * data.ndim
                    for i, m in enumerate(amode):
                        try:
                            if m is not None:
                                tex.set_address_mode(i, address_mode_map[m])
                        except KeyError:
                            raise ValueError('Unknown address mode: ' + m)
                if 'filter_mode' in v and v['filter_mode'] is not None:
                    fmode = v['filter_mode']
                    try:
                        tex.set_filter_mode(filter_mode_map[fmode])
                    except KeyError:
                        raise ValueError('Unknown filter mode: ' + fmode)
                if 'normalized_coordinates' in v and v['normalized_coordinates']:
                    tex.set_flags(tex.get_flags() | drv.TRSF_NORMALIZED_COORDINATES)

        """

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        """runs the CUDA kernel passed as 'func'

        :param func: A cupy kernel compiled for this specific kernel configuration
        :type func: cupy.RawKernel

        :param gpu_args: A list of arguments to the kernel, order should match the
            order in the code. Allowed values are either variables in global memory
            or single values passed by value.
        :type gpu_args: list( cupy.ndarray, numpy.int32, ...)

        :param threads: A tuple listing the number of threads in each dimension of
            the thread block
        :type threads: tuple(int, int, int)

        :param grid: A tuple listing the number of thread blocks in each dimension
            of the grid
        :type grid: tuple(int, int)
        """
        #func(*gpu_args, block=threads, grid=grid, stream=stream, shared=self.smem_size, texrefs=self.texrefs)
        func(grid, threads, gpu_args, stream=stream, shared_mem=self.smem_size)

    def memset(self, allocation, value, size):
        """set the memory in allocation to the value in value

        :param allocation: A GPU memory allocation unit
        :type allocation: cupy.ndarray

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        allocation[:] = value
        #drv.memset_d8(allocation, value, size)


    def memcpy_dtoh(self, dest, src):
        """perform a device to host memory copy

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: cupy.ndarray
        """
        #
        #if isinstance(src, drv.DeviceAllocation):
        #    drv.memcpy_dtoh(dest, src)
        #else:
        #    dest = src
        if isinstance(dest, np.ndarray):
            tmp = cp.asnumpy(src)
            np.copyto(dest, tmp)
        elif isinstance(dest, cp.ndarray):
            cp.copyto(dest, src)
        else:
            raise ValueError("dest type not supported")
        #dest[:] = cp.asnumpy(src)

    def memcpy_htod(self, dest, src):
        """perform a host to device memory copy

        :param dest: A GPU memory allocation unit
        :type dest: cupy.ndarray

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        #if isinstance(dest, drv.DeviceAllocation):
        #    drv.memcpy_htod(dest, src)
        #else:
        #    dest = src
        if isinstance(src, np.ndarray):
            src = cp.asarray(src)
        cp.copyto(dest, src)
        #dest = cp.asarray(src)

    units = {'time': 'ms'}
