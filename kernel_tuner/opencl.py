"""This module contains all OpenCL specific kernel_tuner functions"""
import numpy

#embedded in try block to be able to generate documentation
try:
    import pyopencl as cl
except Exception:
    pass


class OpenCLFunctions(object):
    """Class that groups the OpenCL functions on maintains some state about the device"""

    def __init__(self, device=0):
        #setup context and queue
        platforms = cl.get_platforms()
        self.ctx = cl.Context(dev_type=cl.device_type.ALL,
                properties=[(cl.context_properties.PLATFORM, platforms[device])])
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.mf = cl.mem_flags
        #inspect device properties
        self.max_threads = self.ctx.devices[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE

    def create_gpu_args(self, arguments):
        """ready argument list to be passed to the kernel, allocates gpu mem"""
        gpu_args = []
        for arg in arguments:
            # if arg i is a numpy array copy to device
            if isinstance(arg, numpy.ndarray):
                gpu_args.append(cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=arg))
            else: # if not an array, just pass argument along
                gpu_args.append(arg)
        return gpu_args

    def compile(self, kernel_name, kernel_string):
        """call the CUDA compiler to compile the kernel, return the device function"""
        prg = cl.Program(self.ctx, kernel_string).build()
        func = getattr(prg, kernel_name)
        return func

    def benchmark(self, func, gpu_args, threads, grid):
        """runs the kernel and measures time repeatedly, returns average time"""
        global_size = (grid[0]*threads[0], grid[1]*threads[1], threads[2])
        local_size = threads
        ITERATIONS = 7
        times = []
        for _ in range(ITERATIONS):
            event = func(self.queue, global_size, local_size, *gpu_args)
            event.wait()
            times.append((event.profile.end - event.profile.start)*1e-6)
        times = sorted(times)
        return numpy.mean(times[1:-1])
