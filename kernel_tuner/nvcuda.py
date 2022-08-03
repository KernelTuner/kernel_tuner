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


def error_check(error):
    if isinstance(error, cuda.CUresult):
        if error != cuda.CUresult.CUDA_SUCCESS:
            _, name = cuda.cuGetErrorName(error)
            raise RuntimeError(f"CUDA error: {name.decode()}")
    elif isinstance(error, cudart.cudaError_t):
        if error != cudart.cudaError_t.cudaSuccess:
            _, name = cudart.getErrorName(error)
            raise RuntimeError(f"CUDART error: {name.decode()}")
    elif isinstance(error, nvrtc.nvrtcResult):
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, desc = nvrtc.nvrtcGetErrorString(error)
            raise RuntimeError(f"NVRTC error: {desc.decode()}")
    

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
        error_check(err)
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
        error_check(err)
        err, self.device = cuda.cuDeviceGet(device)
        error_check(err)
        err, self.context = cuda.cuCtxCreate(0, self.device)
        error_check(err)

        # compute capabilities and device properties
        err, major = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device)
        error_check(err)
        err, minor = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device)
        error_check(err)
        err, self.max_threads = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock, device)
        error_check(err)
        self.cc = f"{major}{minor}"
        self.iterations = iterations
        self.current_module = None
        self.func = None
        self.compiler_options = compiler_options or []
        self.compiler_options_bytes = []
        for option in self.compiler_options:
            self.compiler_options_bytes.append(str(option).encode("UTF-8"))

        # create a stream and events
        err, self.stream = cuda.cuStreamCreate(0)
        error_check(err)
        err, self.start = cuda.cuEventCreate(0)
        error_check(err)
        err, self.end = cuda.cuEventCreate(0)
        error_check(err)

        # default dynamically allocated shared memory size, can be overwritten using smem_args
        self.smem_size = 0

        # setup observers
        self.observers = observers or []
        self.observers.append(CudaRuntimeObserver(self))
        for observer in self.observers:
            observer.register_device(self)

        # collect environment information
        err, device_properties = cudart.cudaGetDeviceProperties(device)
        error_check(err)
        env = dict()
        env["device_name"] = device_properties.name.decode()
        env["cuda_version"] = cuda.CUDA_VERSION
        env["compute_capability"] = self.cc
        env["iterations"] = self.iterations
        env["compiler_options"] = self.compiler_options
        env["device_properties"] = str(device_properties).replace("\n", ", ")
        self.env = env
        self.name = env["device_name"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        """free memory and destroy device context"""
        for device_memory in self.allocations:
            if isinstance(device_memory, cuda.CUdeviceptr):
                err = cuda.cuMemFree(device_memory)
                error_check(err)
        if hasattr(self, "context"):
            err = cuda.cuCtxDestroy(self.context)
            error_check(err)

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
                error_check(err)
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

        compiler_options = self.compiler_options_bytes
        if not any([b"--std=" in opt for opt in compiler_options]):
            compiler_options.append(b"--std=c++11")
        if not any(["--std=" in opt for opt in self.compiler_options]):
            self.compiler_options.append("--std=c++11")
        if not any([b"--gpu-architecture=" in opt for opt in compiler_options]):
            compiler_options.append(f"--gpu-architecture=compute_{self.cc}".encode("UTF-8"))
        if not any(["--gpu-architecture=" in opt for opt in self.compiler_options]):
            self.compiler_options.append(f"--gpu-architecture=compute_{self.cc}")

        err, program = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b"CUDAProgram", 0, [], [])
        error_check(err)
        err = nvrtc.nvrtcCompileProgram(program, len(compiler_options), compiler_options)
        error_check(err)
        err, size = nvrtc.nvrtcGetPTXSize(program)
        error_check(err)
        buffer = b' ' * size
        err = nvrtc.nvrtcGetPTX(program, buffer)
        error_check(err)
        err, self.current_module = cuda.cuModuleLoadData(np.char.array(buffer))
        error_check(err)
        err, self.func = cuda.cuModuleGetFunction(self.current_module, str.encode(kernel_name))
        error_check(err)
        return self.func
        
    def start_event(self):
        """ Records the event that marks the start of a measurement """
        err = cudart.cudaEventRecord(self.start, self.stream)
        error_check(err)

    def stop_event(self):
        """ Records the event that marks the end of a measurement """
        err = cudart.cudaEventRecord(self.end, self.stream)
        error_check(err)

    def kernel_finished(self):
        """ Returns True if the kernel has finished, False otherwise """
        err = cudart.cudaEventQuery(self.end)
        if err[0] == cudart.cudaError_t.cudaSuccess:
            return True
        else:
            return False

    def synchronize(self):
        """ Halts execution until device has finished its tasks """
        err = cudart.cudaDeviceSynchronize()
        error_check(err)


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
            err, symbol, _ = cuda.cuModuleGetGlobal(self.current_module, str.encode(k))
            error_check(err)
            err = cuda.cuMemcpyHtoD(symbol, v, v.nbytes)
            error_check(err)

    def copy_shared_memory_args(self, smem_args):
        """add shared memory arguments to the kernel"""
        self.smem_size = smem_args["size"]

    def copy_texture_memory_args(self, texmem_args):
        """adds texture memory arguments to the most recently compiled module

        :param texmem_args: A dictionary containing the data to be passed to the
            device texture memory. See tune_kernel().
        :type texmem_args: dict
        """
        raise NotImplementedError('NVIDIA CUDA backend does not yet support texture memory')

        filter_mode_map = {
            'point': cuda.CUfilter_mode(0),
            'linear': cuda.CUfilter_mode(1)
        }
        address_mode_map = {
            'border': cuda.CUaddress_mode(3),
            'clamp': cuda.CUaddress_mode(1),
            'mirror': cuda.CUaddress_mode(2),
            'wrap': cuda.CUaddress_mode(0)
        }

        self.texrefs = []
        for k, v in texmem_args.items():
            err, tex = cuda.cuModuleGetTexRef(self.current_module, k)
            error_check(err)
            self.texrefs.append(tex)

            if not isinstance(v, dict):
                data = v
            else:
                data = v['array']

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
        arg_types = list()
        for arg in gpu_args:
            if isinstance(arg, cuda.CUdeviceptr):
                arg_types.append(None)
            else:
                arg_types.append(np.ctypeslib.as_ctypes_type(arg.dtype))
        kernel_args  = (tuple(gpu_args), tuple(arg_types))
        err = cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2], threads[0], threads[1], threads[2], self.smem_size, stream, kernel_args, 0)
        error_check(err)

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
        error_check(err)

    def memcpy_dtoh(self, dest, src):
        """perform a device to host memory copy

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: cupy.ndarray
        """
        err = cuda.cuMemcpyDtoH(dest, src, dest.nbytes)
        error_check(err)

    def memcpy_htod(self, dest, src):
        """perform a host to device memory copy

        :param dest: A GPU memory allocation unit
        :type dest: cupy.ndarray

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        err = cuda.cuMemcpyHtoD(dest, src, src.nbytes)
        error_check(err)

    units = {'time': 'ms'}
