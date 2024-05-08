"""This module contains all NVIDIA cuda-python specific kernel_tuner functions."""
from warnings import warn

import numpy as np

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.nvcuda import CudaRuntimeObserver
from kernel_tuner.util import SkippableFailure, cuda_error_check, to_valid_nvrtc_gpu_arch_cc

# embedded in try block to be able to generate documentation
# and run tests without cuda-python installed
try:
    from cuda import cuda, cudart, nvrtc
except ImportError:
    cuda = None


class CudaFunctions(GPUBackend):
    """Class that groups the Cuda functions on maintains state about the device."""

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """Instantiate CudaFunctions object used for interacting with the CUDA device.

        Instantiating this object will inspect and store certain device properties at
        runtime, which are used during compilation and/or execution of kernels by the
        kernel tuner. It also maintains a reference to the most recently compiled
        source module for copying data to constant memory before kernel launch.

        :param device: Number of CUDA device to use for this context
        :type device: int

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int

        :param compiler_options: Compiler options for the CUDA runtime compiler

        :param observers: List of Observer type objects
        """
        self.allocations = []
        self.texrefs = []
        if not cuda:
            raise ImportError(
                "cuda-python not installed, install using 'pip install cuda-python', or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda."
            )

        # initialize and select device
        err = cuda.cuInit(0)
        cuda_error_check(err)
        err, self.device = cuda.cuDeviceGet(device)
        cuda_error_check(err)
        err, self.context = cuda.cuDevicePrimaryCtxRetain(device)
        cuda_error_check(err)
        if CudaFunctions.last_selected_device != device:
            err = cuda.cuCtxSetCurrent(self.context)
            cuda_error_check(err)
            CudaFunctions.last_selected_device = device

        # compute capabilities and device properties
        err, major = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device
        )
        cuda_error_check(err)
        err, minor = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device
        )
        cuda_error_check(err)
        err, self.max_threads = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock, device
        )
        cuda_error_check(err)
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
        cuda_error_check(err)
        err, self.start = cuda.cuEventCreate(0)
        cuda_error_check(err)
        err, self.end = cuda.cuEventCreate(0)
        cuda_error_check(err)

        # default dynamically allocated shared memory size, can be overwritten using smem_args
        self.smem_size = 0

        # setup observers
        self.observers = observers or []
        self.observers.append(CudaRuntimeObserver(self))
        for observer in self.observers:
            observer.register_device(self)

        # collect environment information
        err, device_properties = cudart.cudaGetDeviceProperties(device)
        cuda_error_check(err)
        env = dict()
        env["device_name"] = device_properties.name.decode()
        env["cuda_version"] = cuda.CUDA_VERSION
        env["compute_capability"] = self.cc
        env["iterations"] = self.iterations
        env["compiler_options"] = self.compiler_options
        env["device_properties"] = str(device_properties).replace("\n", ", ")
        self.env = env
        self.name = env["device_name"]

    def __del__(self):
        for device_memory in self.allocations:
            if isinstance(device_memory, cuda.CUdeviceptr):
                err = cuda.cuMemFree(device_memory)
                cuda_error_check(err)

    def ready_argument_list(self, arguments):
        """Ready argument list to be passed to the kernel, allocates gpu mem.

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
                cuda_error_check(err)
                self.allocations.append(device_memory)
                gpu_args.append(device_memory)
                self.memcpy_htod(device_memory, arg)
            # if not array, just pass along
            else:
                gpu_args.append(arg)
        return gpu_args

    def compile(self, kernel_instance):
        """Call the CUDA compiler to compile the kernel, return the device function.

        :param kernel_name: The name of the kernel to be compiled, used to lookup the
            function after compilation.
        :type kernel_name: string

        :param kernel_string: The CUDA kernel code that contains the function `kernel_name`
        :type kernel_string: string

        :returns: A kernel that can be launched by the CUDA runtime
        :rtype:
        """
        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name

        # mimic pycuda behavior to wrap kernel_string in extern "C" if not in kernel_string already
        if 'extern "C"' not in kernel_string:
            kernel_string = 'extern "C" {\n' + kernel_string + "\n}"

        compiler_options = self.compiler_options_bytes
        if not any([b"--std=" in opt for opt in compiler_options]):
            compiler_options.append(b"--std=c++11")
        if not any(["--std=" in opt for opt in self.compiler_options]):
            self.compiler_options.append("--std=c++11")
        if not any([b"--gpu-architecture=" in opt or b"-arch" in opt for opt in compiler_options]):
            compiler_options.append(
                f"--gpu-architecture=compute_{to_valid_nvrtc_gpu_arch_cc(self.cc)}".encode("UTF-8")
            )
        if not any(["--gpu-architecture=" in opt or "-arch" in opt for opt in self.compiler_options]):
            self.compiler_options.append(f"--gpu-architecture=compute_{to_valid_nvrtc_gpu_arch_cc(self.cc)}")

        err, program = nvrtc.nvrtcCreateProgram(
            str.encode(kernel_string), b"CUDAProgram", 0, [], []
        )
        try:
            cuda_error_check(err)
            err = nvrtc.nvrtcCompileProgram(
                program, len(compiler_options), compiler_options
            )
            cuda_error_check(err)
            err, size = nvrtc.nvrtcGetPTXSize(program)
            cuda_error_check(err)
            buff = b" " * size
            err = nvrtc.nvrtcGetPTX(program, buff)
            cuda_error_check(err)
            err, self.current_module = cuda.cuModuleLoadData(np.char.array(buff))
            if err == cuda.CUresult.CUDA_ERROR_INVALID_PTX:
                raise SkippableFailure("uses too much shared data")
            else:
                cuda_error_check(err)
            err, self.func = cuda.cuModuleGetFunction(
                self.current_module, str.encode(kernel_name)
            )
            cuda_error_check(err)

            # get the number of registers per thread used in this kernel
            num_regs = cuda.cuFuncGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func)
            assert num_regs[0] == 0, f"Retrieving number of registers per thread unsuccesful: code {num_regs[0]}"
            self.num_regs = num_regs[1]

        except RuntimeError as re:
            _, n = nvrtc.nvrtcGetProgramLogSize(program)
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(program, log)
            print(log.decode("utf-8"))
            raise re

        return self.func

    def start_event(self):
        """Records the event that marks the start of a measurement."""
        err = cudart.cudaEventRecord(self.start, self.stream)
        cuda_error_check(err)

    def stop_event(self):
        """Records the event that marks the end of a measurement."""
        err = cudart.cudaEventRecord(self.end, self.stream)
        cuda_error_check(err)

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        err = cudart.cudaEventQuery(self.end)
        if err[0] == cudart.cudaError_t.cudaSuccess:
            return True
        else:
            return False

    @staticmethod
    def synchronize():
        """Halts execution until device has finished its tasks."""
        err = cudart.cudaDeviceSynchronize()
        cuda_error_check(err)

    def copy_constant_memory_args(self, cmem_args):
        """Adds constant memory arguments to the most recently compiled module.

        :param cmem_args: A dictionary containing the data to be passed to the
            device constant memory. The format to be used is as follows: A
            string key is used to name the constant memory symbol to which the
            value needs to be copied. Similar to regular arguments, these need
            to be numpy objects, such as numpy.ndarray or numpy.int32, and so on.
        :type cmem_args: dict( string: numpy.ndarray, ... )
        """
        for k, v in cmem_args.items():
            err, symbol, _ = cuda.cuModuleGetGlobal(self.current_module, str.encode(k))
            cuda_error_check(err)
            err = cuda.cuMemcpyHtoD(symbol, v, v.nbytes)
            cuda_error_check(err)

    def copy_shared_memory_args(self, smem_args):
        """Add shared memory arguments to the kernel."""
        self.smem_size = smem_args["size"]

    def copy_texture_memory_args(self, texmem_args):
        """Adds texture memory arguments to the most recently compiled module.

        :param texmem_args: A dictionary containing the data to be passed to the
            device texture memory. See tune_kernel().
        :type texmem_args: dict
        """
        raise NotImplementedError("NVIDIA CUDA backend does not support texture memory")

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        """Runs the CUDA kernel passed as 'func'.

        :param func: A CUDA kernel compiled for this specific kernel configuration
        :type func: cuda.CUfunction

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
        if stream is None:
            stream = self.stream
        arg_types = list()
        for arg in gpu_args:
            if isinstance(arg, cuda.CUdeviceptr):
                arg_types.append(None)
            else:
                arg_types.append(np.ctypeslib.as_ctypes_type(arg.dtype))
        kernel_args = (tuple(gpu_args), tuple(arg_types))
        err = cuda.cuLaunchKernel(
            func,
            grid[0],
            grid[1],
            grid[2],
            threads[0],
            threads[1],
            threads[2],
            self.smem_size,
            stream,
            kernel_args,
            0,
        )
        cuda_error_check(err)

    @staticmethod
    def memset(allocation, value, size):
        """Set the memory in allocation to the value in value.

        :param allocation: A GPU memory allocation unit
        :type allocation: cupy.ndarray

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        err = cudart.cudaMemset(allocation, value, size)
        cuda_error_check(err)

    @staticmethod
    def memcpy_dtoh(dest, src):
        """Perform a device to host memory copy.

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: cuda.CUdeviceptr
        """
        err = cuda.cuMemcpyDtoH(dest, src, dest.nbytes)
        cuda_error_check(err)

    @staticmethod
    def memcpy_htod(dest, src):
        """Perform a host to device memory copy.

        :param dest: A GPU memory allocation unit
        :type dest: cuda.CUdeviceptr

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        err = cuda.cuMemcpyHtoD(dest, src, src.nbytes)
        cuda_error_check(err)

    units = {"time": "ms"}

    last_selected_device = None
