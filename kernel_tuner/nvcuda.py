"""This module contains all NVIDIA cuda-python specific kernel_tuner functions"""
from __future__ import print_function


import logging
import time
import numpy as np

from kernel_tuner.observers import BenchmarkObserver

#embedded in try block to be able to generate documentation
#and run tests without cupy installed
try:
    from cuda import cuda, cudart, nvrtc
except ImportError:
    cuda = None


class CudaRuntimeObserver(BenchmarkObserver):
    """ Observer that measures time using CUDA events during benchmarking """
    def __init__(self, dev):
        self.dev = dev
        self.stream = dev.stream
        self.start = dev.start
        self.end = dev.end
        self.times = []

    def after_finish(self):
        # time in ms
        err, time = cudart.cudaEventElapsedTime(self.start, self.end)
        self.times.append(time)

    def get_results(self):
        results = {
            "time": np.average(self.times),
            "times": self.times.copy()
        }
        self.times = []
        return results


class CudaFunctions:
    """Class that groups the Cuda functions on maintains state about the device"""

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
        if not cuda:
            raise ImportError("Error: cuda-python not installed, please install e.g. " +
                            "using 'pip install cuda-python', please check https://github.com/NVIDIA/cuda-python.")

        # initialize and select device
        err = cuda.cuInit(0)
        self.device = cuda.cuDevice(device)
        err, self.context = cuda.cuCtxCreate(0, self.device)

        # compute capabilities and device properties
        err, major = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device)
        err, minor = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device)
        self.max_threads = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock, device)
        self.cc = f"{major}{minor}"
        self.iterations = iterations
        self.current_module = None
        self.func = None
        self.compiler_options = compiler_options or []

        # create a stream and events
        err, self.stream = cuda.cuStreamCreate(0)
        err, self.start = cuda.cuEventCreate(0)
        err, self.end = cuda.cuEventCreate(0)

        # default dynamically allocated shared memory size, can be overwritten using smem_args
        self.smem_size = 0

        # setup observers
        self.observers = observers or []
        self.observers.append(CudaRuntimeObserver(self))
        for observer in self.observers:
            observer.register_device(self)

        # collect environment information
        err, device_properties = cudart.cudaGetDeviceProperties(device)
        env = dict()
        env["device_name"] = device_properties.name
        env["cuda_version"] = cuda.CUDA_VERSION
        env["compute_capability"] = self.cc
        env["iterations"] = self.iterations
        env["compiler_options"] = self.compiler_options
        env["device_properties"] = device_properties
        self.env = env
        self.name = env["device_name"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        """free memory and destroy device context"""
        for device_memory in self.allocations:
            if isinstance(device_memory, cuda.CUdeviceptr):
                err = cuda.cuMemFree(device_memory)
        if hasattr(self, "context"):
            err = cuda.cuCtxDestroy(self.context)

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
            # if arg is a numpy array copy it to device
            if isinstance(arg, np.ndarray):
                err, device_memory = cuda.cuMemAlloc(arg.nbytes)
                self.allocations.append(device_memory)
                gpu_args.append(device_memory)
                self.memcpy_htod(device_memory, arg)
            # if not array, just pass along
            else:
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
        :rtype: 
        """
        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name

        compiler_options = self.compiler_options
        if not any(["--std=" in opt for opt in compiler_options]):
            compiler_options.append("--std=c++11")
        if not any(["--gpu-architecture="]):
            compiler_options.append(f"--gpu-architecture=compute_{self.cc}")

        err, program = nvrtc.nvrtcCreateProgram(kernel_string, "CUDAProgram", 0, [], [])
        err = nvrtc.nvrtcCompileProgram(program, len(compiler_options, compiler_options))
        err, size = nvrtc.nvrtcGetPTXSize(program)
        buffer = b' ' * size
        err = nvrtc.nvrtcGetPTX(program, buffer)
        err, self.current_module = cuda.cuModuleLoadData(np.char.array(buffer))
        err, func = cuda.cuModuleGetFunction(self.current_module, kernel_name)
        return func
        
    def start_event(self):
        """ Records the event that marks the start of a measurement """
        err = cudart.cudaEventRecord(self.start, self.stream)

    def stop_event(self):
        """ Records the event that marks the end of a measurement """
        err = cudart.cudaEventRecord(self.end, self.stream)

    def kernel_finished(self):
        """ Returns True if the kernel has finished, False otherwise """
        if cudart.cudaEventQuery(self.end) == cuda.CUresult.CUDA_SUCCESS:
            return True

    def synchronize(self):
        """ Halts execution until device has finished its tasks """
        err = cudart.cudaDeviceSynchronize()


    def copy_constant_memory_args(self, cmem_args):
        """adds constant memory arguments to the most recently compiled module

        :param cmem_args: A dictionary containing the data to be passed to the
            device constant memory. The format to be used is as follows: A
            string key is used to name the constant memory symbol to which the
            value needs to be copied. Similar to regular arguments, these need
            to be numpy objects, such as numpy.ndarray or numpy.int32, and so on.
        :type cmem_args: dict( string: numpy.ndarray, ... )
        """
        for k, v in cmem_args.items():
            symbol = cuda.cuModuleGetGlobal(self.current_module, k)
            err = cuda.cuMemcpyHtoD(symbol, v, v.nbytes)

    def copy_shared_memory_args(self, smem_args):
        """add shared memory arguments to the kernel"""
        self.smem_size = smem_args["size"]

    def copy_texture_memory_args(self, texmem_args):
        """adds texture memory arguments to the most recently compiled module

        :param texmem_args: A dictionary containing the data to be passed to the
            device texture memory. See tune_kernel().
        :type texmem_args: dict
        """
        raise NotImplementedError('CUDA backend does not yet support texture memory')

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
        cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2], threads[0], threads[1], threads[2], self.smem_size, stream, gpu_args, 0)

    def memset(self, allocation, value, size):
        """set the memory in allocation to the value in value

        :param allocation: A GPU memory allocation unit
        :type allocation: cupy.ndarray

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        err = cudart.cudaMemset(allocation, value, size)

    def memcpy_dtoh(self, dest, src):
        """perform a device to host memory copy

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: cupy.ndarray
        """
        err = cuda.cuMemcpyDtoH(dest, src, dest.nbytes)

    def memcpy_htod(self, dest, src):
        """perform a host to device memory copy

        :param dest: A GPU memory allocation unit
        :type dest: cupy.ndarray

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        err = cuda.cuMemcpyHtoD(dest, src, src.nbytes)

    units = {'time': 'ms'}
