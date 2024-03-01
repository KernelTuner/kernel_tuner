"""This module contains all Cupy specific kernel_tuner functions."""
from __future__ import print_function
from warnings import warn

import numpy as np

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.cupy import CupyRuntimeObserver

# embedded in try block to be able to generate documentation
# and run tests without cupy installed
try:
    import cupy as cp
except ImportError:
    cp = None


class CupyFunctions(GPUBackend):
    """Class that groups the Cupy functions on maintains state about the device."""

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """Instantiate CupyFunctions object used for interacting with the CUDA device.

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
            raise ImportError(
                "cupy not installed, install using 'pip install cupy', or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda."
            )

        # select device
        self.dev = dev = cp.cuda.Device(device)
        dev.use()

        # inspect device properties
        self.devprops = dev.attributes
        self.cc = dev.compute_capability
        self.max_threads = self.devprops["MaxThreadsPerBlock"]

        self.iterations = iterations
        self.current_module = None
        self.func = None
        self.compiler_options = compiler_options or []

        # create a stream and events
        self.stream = cp.cuda.Stream()
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()

        # default dynamically allocated shared memory size, can be overwritten using smem_args
        self.smem_size = 0

        # setup observers
        self.observers = observers or []
        self.observers.append(CupyRuntimeObserver(self))
        for obs in self.observers:
            obs.register_device(self)

        # collect environment information
        env = dict()
        cupy_info = str(cp._cupyx.get_runtime_info()).split("\n")[:-1]
        info_dict = {
            s.split(":")[0].strip(): s.split(":")[1].strip() for s in cupy_info
        }
        env["device_name"] = info_dict[f"Device {device} Name"]

        env["cuda_version"] = cp.cuda.runtime.driverGetVersion()
        env["compute_capability"] = self.cc
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        env["device_properties"] = self.devprops
        self.env = env
        self.name = env["device_name"]

    def ready_argument_list(self, arguments):
        """Ready argument list to be passed to the kernel, allocates gpu mem.

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
            # if not a numpy array, just pass argument along
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

        :returns: An CUDA kernel that can be called directly.
        :rtype: cupy.RawKernel
        """
        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name

        compiler_options = self.compiler_options
        if not any(["-std=" in opt for opt in self.compiler_options]):
            compiler_options = ["--std=c++11"] + self.compiler_options
        # CuPy already sets the --gpu-architecture by itself, as per https://github.com/cupy/cupy/blob/main/cupy/cuda/compiler.py#L145

        options = tuple(compiler_options)

        self.current_module = cp.RawModule(
            code=kernel_string, options=options, name_expressions=[kernel_name]
        )

        self.func = self.current_module.get_function(kernel_name)
        self.num_regs = self.func.num_regs
        return self.func

    def start_event(self):
        """Records the event that marks the start of a measurement."""
        self.start.record(stream=self.stream)

    def stop_event(self):
        """Records the event that marks the end of a measurement."""
        self.end.record(stream=self.stream)

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        return self.end.done

    def synchronize(self):
        """Halts execution until device has finished its tasks."""
        self.dev.synchronize()

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
            symbol = self.current_module.get_global(k)
            constant_mem = cp.ndarray(v.shape, v.dtype, symbol)
            constant_mem[:] = cp.asarray(v)

    def copy_shared_memory_args(self, smem_args):
        """Add shared memory arguments to the kernel."""
        self.smem_size = smem_args["size"]

    def copy_texture_memory_args(self, texmem_args):
        """Adds texture memory arguments to the most recently compiled module.

        :param texmem_args: A dictionary containing the data to be passed to the
            device texture memory. See tune_kernel().
        :type texmem_args: dict
        """
        raise NotImplementedError("CuPy backend does not support texture memory")

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        """Runs the CUDA kernel passed as 'func'.

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
        if stream is None:
            stream = self.stream
        func(grid, threads, gpu_args, stream=stream, shared_mem=self.smem_size)

    def memset(self, allocation, value, size):
        """Set the memory in allocation to the value in value.

        :param allocation: A GPU memory allocation unit
        :type allocation: cupy.ndarray

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        allocation[:] = value

    def memcpy_dtoh(self, dest, src):
        """Perform a device to host memory copy.

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: cupy.ndarray
        """
        if isinstance(dest, np.ndarray):
            tmp = cp.asnumpy(src)
            np.copyto(dest, tmp)
        elif isinstance(dest, cp.ndarray):
            cp.copyto(dest, src)
        else:
            raise ValueError("dest type not supported")

    def memcpy_htod(self, dest, src):
        """Perform a host to device memory copy.

        :param dest: A GPU memory allocation unit
        :type dest: cupy.ndarray

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        if isinstance(src, np.ndarray):
            src = cp.asarray(src)
        cp.copyto(dest, src)

    units = {"time": "ms"}
