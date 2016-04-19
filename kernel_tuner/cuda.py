"""This module contains all CUDA specific kernel_tuner functions"""
import numpy

#embedded in try block to be able to generate documentation
#and run tests without pycuda installed
try:
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
except Exception:
    drv = object()
    SourceModule = object()



class CudaFunctions(object):
    """Class that groups the CUDA functions on maintains some state about the device"""

    def __init__(self, device=0):
        drv.init()
        self.context = drv.Device(device).make_context()

        #inspect device properties
        devprops = { str(k): v for (k, v) in self.context.get_device().get_attributes().items() }
        self.max_threads = devprops['MAX_THREADS_PER_BLOCK']
        self.cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])
        self.ITERATIONS = 7
        self.current_module = None

    def __del__(self):
        self.context.pop()


    def create_gpu_args(self, arguments):
        """ready argument list to be passed to the kernel, allocates gpu mem"""
        gpu_args = []
        for arg in arguments:
            # if arg i is a numpy array copy to device
            if isinstance(arg, numpy.ndarray):
                gpu_args.append(drv.mem_alloc(arg.nbytes))
                drv.memcpy_htod(gpu_args[-1], arg)
            else: # if not an array, just pass argument along
                gpu_args.append(arg)
        return gpu_args


    def compile(self, kernel_name, kernel_string):
        """call the CUDA compiler to compile the kernel, return the device function"""
        try:
            self.current_module = SourceModule(kernel_string, options=['-Xcompiler=-Wall'],
                    arch='compute_' + self.cc, code='sm_' + self.cc,
                    cache_dir=False)
            func = self.current_module.get_function(kernel_name)
            return func
        except drv.CompileError, e:
            if "uses too much shared data" in e.stderr:
                raise Exception("uses too much shared data")
            else:
                raise e

    def benchmark(self, func, gpu_args, threads, grid):
        """runs the kernel and measures time repeatedly, returns average time"""
        start = drv.Event()
        end = drv.Event()
        times = []
        for _ in range(self.ITERATIONS):
            self.context.synchronize()
            start.record()
            func(*gpu_args, block=threads, grid=grid)
            end.record()
            self.context.synchronize()
            times.append(end.time_since(start))
        times = sorted(times)
        return numpy.mean(times[1:-1])

    def copy_constant_memory_args(self, cmem_args):
        """adds constant memory arguments to the most recently compiled module"""
        for k,v in cmem_args.iteritems():
            symbol = self.current_module.get_global(k)[0]
            drv.memcpy_htod(symbol, v)
