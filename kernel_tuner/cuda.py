"""This module contains all CUDA specific kernel_tuner functions"""
from __future__ import print_function

import logging
import numpy

#embedded in try block to be able to generate documentation
#and run tests without pycuda installed
try:
    import pycuda.driver as drv
except ImportError:
    drv = None
try:
    from pycuda.compiler import SourceModule
except ImportError:
    SourceModule = None
try:
    from pycuda.compiler import DynamicSourceModule
except ImportError:
    DynamicSourceModule = None


class CudaFunctions(object):
    """Class that groups the CUDA functions on maintains state about the device"""

    def __init__(self, device=0, iterations=7, compiler_options=None):
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
        if not drv:
            raise ImportError("Error: pycuda not installed, please install e.g. using 'pip install pycuda'.")

        drv.init()
        self.context = drv.Device(device).make_context()

        #inspect device properties
        devprops = {str(k): v for (k, v) in self.context.get_device().get_attributes().items()}
        self.max_threads = devprops['MAX_THREADS_PER_BLOCK']
        cc = str(devprops.get('COMPUTE_CAPABILITY_MAJOR', '0')) + str(devprops.get('COMPUTE_CAPABILITY_MINOR', '0'))
        if cc == "00":
            cc = self.context.get_device().compute_capability()
        self.cc = str(cc[0])+str(cc[1])
        self.iterations = iterations
        self.current_module = None
        self.compiler_options = compiler_options or []

        #select PyCUDA source module
        if int(self.cc) >= 35:
            self.source_mod = DynamicSourceModule
        else:
            self.source_mod = SourceModule
        if not self.source_mod:
            raise ImportError("Error: pycuda not correctly installed, please ensure pycuda is installed on the same CUDA installation as you're using right now")

        #collect environment information
        env = dict()
        env["device_name"] = self.context.get_device().name()
        env["cuda_version"] = ".".join([str(i) for i in drv.get_version()])
        env["compute_capability"] = self.cc
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        env["device_properties"] = devprops
        self.env = env
        self.name = env["device_name"]

    def __del__(self):
        for gpu_mem in self.allocations:
            if hasattr(gpu_mem, 'free'): #if needed for when using mocks during testing
                gpu_mem.free()
        if hasattr(self, 'context'):
            self.context.pop()

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the kernel, allocates gpu mem

        :param arguments: List of arguments to be passed to the kernel.
            The order should match the argument list on the CUDA kernel.
            Allowed values are numpy.ndarray, and/or numpy.int32, numpy.float32, and so on.
        :type arguments: list(numpy objects)

        :returns: A list of arguments that can be passed to an CUDA kernel.
        :rtype: list( pycuda.driver.DeviceAllocation, numpy.int32, ... )
        """
        gpu_args = []
        for arg in arguments:
            # if arg i is a numpy array copy to device
            if isinstance(arg, numpy.ndarray):
                alloc = drv.mem_alloc(arg.nbytes)
                self.allocations.append(alloc)
                gpu_args.append(alloc)
                drv.memcpy_htod(gpu_args[-1], arg)
            else: # if not an array, just pass argument along
                gpu_args.append(arg)
        return gpu_args


    def compile(self, kernel_name, kernel_string):
        """call the CUDA compiler to compile the kernel, return the device function

        :param kernel_name: The name of the kernel to be compiled, used to lookup the
            function after compilation.
        :type kernel_name: string

        :param kernel_string: The CUDA kernel code that contains the function `kernel_name`
        :type kernel_string: string

        :returns: An CUDA kernel that can be called directly.
        :rtype: pycuda.driver.Function
        """
        try:
            no_extern_c = 'extern "C"' in kernel_string

            compiler_options = ['-Xcompiler=-Wall']
            if self.compiler_options:
                compiler_options += self.compiler_options

            self.current_module = self.source_mod(kernel_string, options=compiler_options + ["-e", kernel_name],
                                             arch=('compute_' + self.cc) if self.cc != "00" else None,
                                             code=('sm_' + self.cc) if self.cc != "00" else None,
                                             cache_dir=False, no_extern_c=no_extern_c)

            func = self.current_module.get_function(kernel_name)
            return func
        except drv.CompileError as e:
            if "uses too much shared data" in e.stderr:
                raise Exception("uses too much shared data")
            else:
                raise e


    def benchmark(self, func, gpu_args, threads, grid, times):
        """runs the kernel and measures time repeatedly, returns average time

        Runs the kernel and measures kernel execution time repeatedly, number of
        iterations is set during the creation of CudaFunctions. Benchmark returns
        a robust average, from all measurements the fastest and slowest runs are
        discarded and the rest is included in the returned average. The reason for
        this is to be robust against initialization artifacts and other exceptional
        cases.

        :param func: A PyCuda kernel compiled for this specific kernel configuration
        :type func: pycuda.driver.Function

        :param gpu_args: A list of arguments to the kernel, order should match the
            order in the code. Allowed values are either variables in global memory
            or single values passed by value.
        :type gpu_args: list( pycuda.driver.DeviceAllocation, numpy.int32, ...)

        :param threads: A tuple listing the number of threads in each dimension of
            the thread block
        :type threads: tuple(int, int, int)

        :param grid: A tuple listing the number of thread blocks in each dimension
            of the grid
        :type grid: tuple(int, int)

        :param times: Return the execution time of all iterations.
        :type times: bool

        :returns: All execution times, if times=True, or a robust average for the
            kernel execution time.
        :rtype: float
        """
        start = drv.Event()
        end = drv.Event()
        time = []
        for _ in range(self.iterations):
            self.context.synchronize()
            start.record()
            self.run_kernel(func, gpu_args, threads, grid)
            end.record()
            self.context.synchronize()
            time.append(end.time_since(start))
        time = sorted(time)
        if times:
            return time
        else:
            if self.iterations > 4:
                return numpy.mean(time[1:-1])
            else:
                return numpy.mean(time)

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
        for k, v in cmem_args.items():
            symbol = self.current_module.get_global(k)[0]
            logging.debug('copying to symbol: ' + str(symbol))
            logging.debug('array to be copied: ')
            logging.debug(v.nbytes)
            logging.debug(v.dtype)
            logging.debug(v.flags)
            drv.memcpy_htod(symbol, v)

    def run_kernel(self, func, gpu_args, threads, grid):
        """runs the CUDA kernel passed as 'func'

        :param func: A PyCuda kernel compiled for this specific kernel configuration
        :type func: pycuda.driver.Function

        :param gpu_args: A list of arguments to the kernel, order should match the
            order in the code. Allowed values are either variables in global memory
            or single values passed by value.
        :type gpu_args: list( pycuda.driver.DeviceAllocation, numpy.int32, ...)

        :param threads: A tuple listing the number of threads in each dimension of
            the thread block
        :type threads: tuple(int, int, int)

        :param grid: A tuple listing the number of thread blocks in each dimension
            of the grid
        :type grid: tuple(int, int)
        """
        func(*gpu_args, block=threads, grid=grid)

    def memset(self, allocation, value, size):
        """set the memory in allocation to the value in value

        :param allocation: A GPU memory allocation unit
        :type allocation: pycuda.driver.DeviceAllocation

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        drv.memset_d8(allocation, int(value), int(size))

    def memcpy_dtoh(self, dest, src):
        """perform a device to host memory copy

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: pycuda.driver.DeviceAllocation
        """
        if isinstance(src, drv.DeviceAllocation):
            drv.memcpy_dtoh(dest, src)
        else:
            dest = src

    units = {'time': 'ms'}
