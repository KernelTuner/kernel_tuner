"""This module contains all OpenCL specific kernel_tuner functions."""
from __future__ import print_function

import numpy as np

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.opencl import OpenCLObserver

# embedded in try block to be able to generate documentation
try:
    import pyopencl as cl
except ImportError:
    cl = None


class OpenCLFunctions(GPUBackend):
    """Class that groups the OpenCL functions on maintains some state about the device."""

    def __init__(
        self, device=0, platform=0, iterations=7, compiler_options=None, observers=None
    ):
        """Creates OpenCL device context and reads device properties.

        :param device: The ID of the OpenCL device to use for benchmarking
        :type device: int

        :param iterations: The number of iterations to run the kernel during benchmarking, 7 by default.
        :type iterations: int
        """
        if not cl:
            raise ImportError(
                "pyopencl not installed, install using 'pip install pyopencl', or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#opencl-and-pyopencl."
            )

        self.iterations = iterations
        # setup context and queue
        platforms = cl.get_platforms()
        self.ctx = cl.Context(devices=[platforms[platform].get_devices()[device]])

        self.queue = cl.CommandQueue(
            self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
        )
        self.mf = cl.mem_flags
        # inspect device properties
        self.max_threads = self.ctx.devices[0].get_info(
            cl.device_info.MAX_WORK_GROUP_SIZE
        )
        self.compiler_options = compiler_options or []

        # observer stuff
        self.observers = observers or []
        self.observers.append(OpenCLObserver(self))
        self.event = None
        for obs in self.observers:
            obs.register_device(self)

        # collect environment information
        dev = self.ctx.devices[0]
        env = dict()
        env["platform_name"] = dev.platform.name
        env["platform_version"] = dev.platform.version
        env["device_name"] = dev.name
        env["device_version"] = dev.version
        env["opencl_c_version"] = dev.opencl_c_version
        env["driver_version"] = dev.driver_version
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        self.env = env
        self.name = dev.name

    def ready_argument_list(self, arguments):
        """Ready argument list to be passed to the kernel, allocates gpu mem.

        :param arguments: List of arguments to be passed to the kernel.
            The order should match the argument list on the OpenCL kernel.
            Allowed values are numpy.ndarray, and/or numpy.int32, numpy.float32, and so on.
        :type arguments: list(numpy objects)

        :returns: A list of arguments that can be passed to an OpenCL kernel.
        :rtype: list( pyopencl.Buffer, numpy.int32, ... )
        """
        gpu_args = []
        for arg in arguments:
            # if arg i is a numpy array copy to device
            if isinstance(arg, np.ndarray):
                gpu_args.append(
                    cl.Buffer(
                        self.ctx,
                        self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                        hostbuf=arg,
                    )
                )
            # if not an array, just pass argument along
            else:
                gpu_args.append(arg)
        return gpu_args

    def compile(self, kernel_instance):
        """Call the OpenCL compiler to compile the kernel, return the device function.

        :param kernel_name: The name of the kernel to be compiled, used to lookup the
            function after compilation.
        :type kernel_name: string

        :param kernel_string: The OpenCL kernel code that contains the function `kernel_name`
        :type kernel_string: string

        :returns: An OpenCL kernel that can be called directly.
        :rtype: pyopencl.Kernel
        """
        prg = cl.Program(self.ctx, kernel_instance.kernel_string).build(
            options=self.compiler_options
        )
        func = getattr(prg, kernel_instance.name)
        return func

    def start_event(self):
        """Records the event that marks the start of a measurement.

        In OpenCL the event is created when the kernel is launched
        """
        pass

    def stop_event(self):
        """Records the event that marks the end of a measurement.

        In OpenCL the event is created when the kernel is launched
        """
        pass

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        return self.event.get_info(cl.event_info.COMMAND_EXECUTION_STATUS) == 0

    def synchronize(self):
        """Halts execution until device has finished its tasks."""
        self.queue.finish()

    def run_kernel(self, func, gpu_args, threads, grid):
        """Runs the OpenCL kernel passed as 'func'.

        :param func: An OpenCL Kernel
        :type func: pyopencl.Kernel

        :param gpu_args: A list of arguments to the kernel, order should match the
            order in the code. Allowed values are either variables in global memory
            or single values passed by value.
        :type gpu_args: list( pyopencl.Buffer, numpy.int32, ...)

        :param threads: A tuple listing the number of work items in each dimension of
            the work group.
        :type threads: tuple(int, int, int)

        :param grid: A tuple listing the number of work groups in each dimension
            of the NDRange.
        :type grid: tuple(int, int)
        """
        global_size = (grid[0] * threads[0], grid[1] * threads[1], grid[2] * threads[2])
        local_size = threads
        self.event = func(self.queue, global_size, local_size, *gpu_args)

    def memset(self, buffer, value, size):
        """Set the memory in allocation to the value in value.

        :param allocation: An OpenCL Buffer to fill
        :type allocation: pyopencl.Buffer

        :param value: The value to set the memory to
        :type value: a single 32-bit int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        if isinstance(buffer, cl.Buffer):
            try:
                cl.enqueue_fill_buffer(self.queue, buffer, np.uint32(value), 0, size)
            except AttributeError:
                src = np.zeros(size, dtype="uint8") + np.uint8(value)
                cl.enqueue_copy(self.queue, buffer, src)

    def memcpy_dtoh(self, dest, src):
        """Perform a device to host memory copy.

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: An OpenCL Buffer to copy data from
        :type src: pyopencl.Buffer
        """
        if isinstance(src, cl.Buffer):
            cl.enqueue_copy(self.queue, dest, src)

    def memcpy_htod(self, dest, src):
        """Perform a host to device memory copy.

        :param dest: An OpenCL Buffer to copy data from
        :type dest: pyopencl.Buffer

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        if isinstance(dest, cl.Buffer):
            cl.enqueue_copy(self.queue, dest, src)

    def copy_constant_memory_args(self, cmem_args):
        raise NotImplementedError("PyOpenCL backend does not support constant memory")

    def copy_shared_memory_args(self, smem_args):
        raise NotImplementedError("PyOpenCL backend does not support shared memory")

    def copy_texture_memory_args(self, texmem_args):
        raise NotImplementedError("PyOpenCL backend does not support texture memory")

    units = {"time": "ms"}
