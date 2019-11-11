"""This module contains all OpenCL specific kernel_tuner functions"""
from __future__ import print_function
import time
import numpy

from kernel_tuner.nvml import nvml

#embedded in try block to be able to generate documentation
try:
    import pyopencl as cl
except ImportError:
    cl = None

#check if power_sensor is installed
try:
    import power_sensor
except ImportError:
    power_sensor = None



class OpenCLFunctions(object):
    """Class that groups the OpenCL functions on maintains some state about the device"""

    def __init__(self, device=0, platform=0, iterations=7, compiler_options=None):
        """Creates OpenCL device context and reads device properties

        :param device: The ID of the OpenCL device to use for benchmarking
        :type device: int

        :param iterations: The number of iterations to run the kernel during benchmarking, 7 by default.
        :type iterations: int
        """
        if not cl:
            raise ImportError("Error: pyopencl not installed, please install e.g. using 'pip install pyopencl'.")

        self.iterations = iterations
        #setup context and queue
        platforms = cl.get_platforms()
        self.ctx = cl.Context(devices=[platforms[platform].get_devices()[device]])

        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.mf = cl.mem_flags
        #inspect device properties
        self.max_threads = self.ctx.devices[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        self.compiler_options = compiler_options or []

        #setup PowerSensor if available
        if power_sensor:
            self.ps = power_sensor.PowerSensor("/dev/ttyACM0")
        else:
            self.ps = None

        #collect environment information
        dev = self.ctx.devices[0]
        env = dict()
        env["vendor_name"] = dev.vendor
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

        if "NVIDIA" in dev.vendor:
            try:
                self.nvml = nvml(device)
                self.use_nvml = True
            except:
                self.use_nvml = False


    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the kernel, allocates gpu mem

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
            if isinstance(arg, numpy.ndarray):
                gpu_args.append(cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=arg))
            else: # if not an array, just pass argument along
                gpu_args.append(arg)
        return gpu_args

    def compile(self, kernel_name, kernel_string):
        """call the OpenCL compiler to compile the kernel, return the device function

        :param kernel_name: The name of the kernel to be compiled, used to lookup the
            function after compilation.
        :type kernel_name: string

        :param kernel_string: The OpenCL kernel code that contains the function `kernel_name`
        :type kernel_string: string

        :returns: An OpenCL kernel that can be called directly.
        :rtype: pyopencl.Kernel
        """
        prg = cl.Program(self.ctx, kernel_string).build(options=self.compiler_options)
        func = getattr(prg, kernel_name)
        return func

    def benchmark(self, func, gpu_args, threads, grid):
        """runs the kernel and measures time repeatedly, returns average time

        Runs the kernel and measures kernel execution time repeatedly, number of
        iterations is set during the creation of OpenCLFunctions. Benchmark returns
        a robust average, from all measurements the fastest and slowest runs are
        discarded and the rest is included in the returned average. The reason for
        this is to be robust against initialization artifacts and other exceptional
        cases.

        :param func: A PyOpenCL kernel compiled for this specific kernel configuration
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

        :returns: All benchmark results.
        :rtype: dict()
        """
        result = dict()
        result["times"] = []
        power = []
        energy = []
        global_size = (grid[0]*threads[0], grid[1]*threads[1], grid[2]*threads[2])
        local_size = threads
        time = []
        for _ in range(self.iterations):
            event = func(self.queue, global_size, local_size, *gpu_args)
            if self.ps:
                begin_state = self.ps.read()
            if self.ps:
                event.wait()
                end_state = self.ps.read()
                ps_measured_t = end_state.time_at_read - begin_state.time_at_read
                cl_measured_t = (event.profile.end - event.profile.start) * 1e-9
                ps_measured_e = power_sensor.Joules(begin_state, end_state, -1) * (cl_measured_t / ps_measured_t)
                energy.append(ps_measured_e)
            elif self.use_nvml:
                energy_consumed, power_readings = self._measure_nvml(event)
                power.append(power_readings) #time in s, power usage in milliwatts
                energy.append(energy_consumed)

            result["times"].append((event.profile.end - event.profile.start)*1e-6)

        if energy:
            result["energies"] = energy
            result["energy"] = numpy.mean(result["energies"])
        result["time"] = numpy.mean(result["times"])
        return result

    def _measure_nvml(self, event):
        #measure power usage until kernel is done
        power_readings = []
        energy = False
        t0 = time.time()
        while event.get_info(cl.event_info.COMMAND_EXECUTION_STATUS) != 0:
            power_readings.append([time.time()-t0, self.nvml.pwr_usage()])
        event.wait()
        execution_time_s = (event.profile.end - event.profile.start)*1e-9 # s

        #pre and postfix to start at 0 and end at kernel end
        if power_readings:
            #prefix to start at t0
            power_readings = [[0.0, power_readings[0][1]]] + power_readings

            #postfix if kernel ended later than our final measurement
            if execution_time_s > power_readings[-1][0]:
                power_readings = power_readings + [[execution_time_s, power_readings[-1][1]]]
            else:
                power_readings[-1][0] = execution_time_s

            #compute energy consumption as area under curve
            x = [d[0] for d in power_readings] #s
            y = [d[1]/1000.0 for d in power_readings] #convert to watts
            energy = numpy.trapz(y,x) #in Joule

        return energy, power_readings



    def run_kernel(self, func, gpu_args, threads, grid):
        """runs the OpenCL kernel passed as 'func'

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
        global_size = (grid[0]*threads[0], grid[1]*threads[1], grid[2]*threads[2])
        local_size = threads
        event = func(self.queue, global_size, local_size, *gpu_args)
        event.wait()

    def memset(self, buffer, value, size):
        """set the memory in allocation to the value in value

        :param allocation: An OpenCL Buffer to fill
        :type allocation: pyopencl.Buffer

        :param value: The value to set the memory to
        :type value: a single 32-bit int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        if isinstance(buffer, cl.Buffer):
            try:
                cl.enqueue_fill_buffer(self.queue, buffer, numpy.uint32(value), 0, size)
            except AttributeError:
                src=numpy.zeros(size, dtype='uint8')+numpy.uint8(value)
                cl.enqueue_copy(self.queue, buffer, src)

    def memcpy_dtoh(self, dest, src):
        """perform a device to host memory copy

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: An OpenCL Buffer to copy data from
        :type src: pyopencl.Buffer
        """
        if isinstance(src, cl.Buffer):
            cl.enqueue_copy(self.queue, dest, src)

    def memcpy_htod(self, dest, src):
        """perform a host to device memory copy

        :param dest: An OpenCL Buffer to copy data from
        :type dest: pyopencl.Buffer

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        if isinstance(dest, cl.Buffer):
            cl.enqueue_copy(self.queue, dest, src)

    units = {'time': 'ms'}
