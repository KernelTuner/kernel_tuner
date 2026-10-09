"""This module contains all CUDA specific kernel_tuner functions."""

from __future__ import print_function

import logging
import threading

import numpy as np

from kernel_tuner.backends.backend import GPUBackend, get_device_array, is_host_array
from kernel_tuner.observers.nvml import nvml  # noqa F401
from kernel_tuner.observers.pycuda import PyCudaRuntimeObserver
from kernel_tuner.util import SHARED_MEMORY_ERROR, SkippableFailure

# embedded in try block to be able to generate documentation
# and run tests without pycuda installed
try:
    import pycuda.driver as drv

    pycuda_available = True
except ImportError:

    class PyCudaPlaceHolder:
        pass

    drv = PyCudaPlaceHolder()
    pycuda_available = False

try:
    from pycuda.compiler import SourceModule
except ImportError:
    SourceModule = None
try:
    from pycuda.compiler import DynamicSourceModule
except ImportError:
    DynamicSourceModule = None


class PyCudaFunctions(GPUBackend):
    """Class that groups the CUDA functions on maintains state about the device."""

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """Instantiate PyCudaFunctions object used for interacting with the CUDA device.

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
        # if not PyCuda available, check if mocking before raising exception
        if not pycuda_available and isinstance(drv, PyCudaPlaceHolder):
            raise ImportError(
                "pycuda not installed, install using 'pip install pycuda', or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda."
            )

        drv.init()
        self.context = drv.Device(device).retain_primary_context()
        if PyCudaFunctions.last_selected_device != device:
            # pycuda does not wrap cuCtxSetCurrent.
            # As an approximation we push the new device's primary context
            # when switching to a different device.
            if PyCudaFunctions.last_selected_context is not None:
                PyCudaFunctions.last_selected_context.pop()
            else:
                import atexit

                def _finish_up():
                    PyCudaFunctions.last_selected_context.pop()

                atexit.register(_finish_up)
            self.context.push()
            PyCudaFunctions.last_selected_device = device
            PyCudaFunctions.last_selected_context = self.context

        # inspect device properties
        devprops = {str(k): v for (k, v) in self.context.get_device().get_attributes().items()}
        self.max_threads = devprops["MAX_THREADS_PER_BLOCK"]
        cc = str(devprops.get("COMPUTE_CAPABILITY_MAJOR", "0")) + str(devprops.get("COMPUTE_CAPABILITY_MINOR", "0"))
        if cc == "00":
            cc = self.context.get_device().compute_capability()
        self.cc = str(cc)
        self.iterations = iterations
        self.current_module = None
        self.func = None
        self.compiler_options = compiler_options or []

        # select PyCUDA source module
        if int(self.cc) >= 35:
            self.source_mod = DynamicSourceModule
        else:
            self.source_mod = SourceModule
        if not self.source_mod:
            raise ImportError(
                "Error: pycuda not correctly installed, please ensure pycuda is installed on the same CUDA installation as you're using right now"
            )

        # create a stream and events
        self.stream = drv.Stream()
        self.start = drv.Event()
        self.end = drv.Event()

        # default dynamically allocated shared memory size, can be overwritten using smem_args
        self.smem_size = 0

        # collect environment information
        env = dict()
        env["device_name"] = self.context.get_device().name()
        env["pci_bus_id"] = self.context.get_device().pci_bus_id()
        env["cuda_version"] = ".".join([str(i) for i in drv.get_version()])
        env["compute_capability"] = self.cc
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        env["device_properties"] = devprops
        self.env = env
        self.name = env["device_name"]

        # setup observers
        self.observers = observers or []
        self.observers.append(PyCudaRuntimeObserver(self))
        for obs in self.observers:
            obs.register_device(self)

    def __del__(self):
        for gpu_mem in self.allocations:
            # if needed for when using mocks during testing
            if hasattr(gpu_mem, "free"):
                gpu_mem.free()

    def ready_argument_list(self, arguments):
        """Ready argument list to be passed to the kernel, allocates gpu mem.

        :param arguments: List of arguments to be passed to the kernel.
            The order should match the argument list on the CUDA kernel.
            Allowed values are numpy.ndarray, and/or numpy.int32, numpy.float32, and so on.
            Device arrays that implement the CUDA Array Interface (e.g. PyTorch CUDA tensors,
            CuPy arrays) and CPU PyTorch tensors are also supported.
        :type arguments: list(numpy objects)

        :returns: A list of arguments that can be passed to an CUDA kernel.
        :rtype: list( pycuda.driver.DeviceAllocation, numpy.int32, ... )
        """
        gpu_args = []
        for arg in arguments:
            # arrays are copied to a new device allocation, so the kernel never modifies the user's data
            device_array = get_device_array(arg)
            if device_array is not None or is_host_array(arg):
                nbytes = device_array[1] if device_array is not None else np.asarray(arg).nbytes
                alloc = drv.mem_alloc(nbytes)
                self.allocations.append(alloc)
                gpu_args.append(alloc)
                self.memcpy_htod(alloc, arg)
            # pycuda does not support bool, convert to uint8 instead
            elif isinstance(arg, np.bool_):
                gpu_args.append(arg.astype(np.uint8))
            # pycuda does not support 16-bit formats, view them as uint16
            elif isinstance(arg, np.generic) and str(arg.dtype) in ("float16", "bfloat16"):
                gpu_args.append(arg.view(np.uint16))
            # if not an array, just pass argument along
            else:
                gpu_args.append(arg)
        return gpu_args

    def compile(self, kernel_instance):
        """Call the CUDA compiler to compile the kernel, return the device function.

        :param kernel_instance: The kernel instance, containing the name and code of the kernel
        :type kernel_instance: kernel_tuner.core.KernelInstance

        :returns: An CUDA kernel that can be called directly.
        :rtype: pycuda.driver.Function
        """
        return self.load(kernel_instance, self.build(kernel_instance))

    def build(self, kernel_instance):
        """Compile the kernel with nvcc and load it into a module, this is thread-safe.

        PyCUDA links and loads the module as part of compilation, so the context of this backend
        is made current for the calling thread while building the module.

        :param kernel_instance: The kernel instance, containing the name and code of the kernel
        :type kernel_instance: kernel_tuner.core.KernelInstance

        :returns: The compiled module
        :rtype: pycuda.compiler.SourceModule
        """
        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name

        no_extern_c = 'extern "C"' in kernel_string

        compiler_options = ["-Xcompiler=-Wall"]
        if self.compiler_options:
            compiler_options += self.compiler_options

        other_thread = threading.current_thread() is not threading.main_thread()
        if other_thread:
            self.context.push()
        try:
            return self.source_mod(
                kernel_string,
                options=compiler_options + ["-e", kernel_name],
                arch=("compute_" + self.cc) if self.cc != "00" else None,
                code=("sm_" + self.cc) if self.cc != "00" else None,
                cache_dir=False,
                no_extern_c=no_extern_c,
            )
        except drv.CompileError as e:
            if SHARED_MEMORY_ERROR in e.stderr:
                raise SkippableFailure(SHARED_MEMORY_ERROR)
            else:
                raise e
        except drv.LogicError as e:
            # DynamicSourceModule compiles to PTX, the PTX is compiled to machine code when it is linked,
            # at which point the CUDA driver rejects kernels that use too much shared memory
            if "cuLinkComplete failed: device kernel image is invalid" in str(e):
                raise SkippableFailure(SHARED_MEMORY_ERROR)
            else:
                raise e
        finally:
            if other_thread:
                drv.Context.pop()

    def load(self, kernel_instance, build_result):
        """Return the device function from a module compiled by build().

        :param kernel_instance: The kernel instance
        :type kernel_instance: kernel_tuner.core.KernelInstance

        :param build_result: The module returned by build()
        :type build_result: pycuda.compiler.SourceModule

        :returns: An CUDA kernel that can be called directly.
        :rtype: pycuda.driver.Function
        """
        self.current_module = build_result
        self.func = self.current_module.get_function(kernel_instance.name)
        if not isinstance(self.func, str):
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
        return self.end.query()

    def synchronize(self):
        """Halts execution until device has finished its tasks."""
        self.context.synchronize()

    def copy_constant_memory_args(self, cmem_args):
        """Adds constant memory arguments to the most recently compiled module.

        :param cmem_args: A dictionary containing the data to be passed to the
            device constant memory. The format to be used is as follows: A
            string key is used to name the constant memory symbol to which the
            value needs to be copied. Similar to regular arguments, these need
            to be numpy objects, such as numpy.ndarray or numpy.int32, and so on.
        :type cmem_args: dict( string: numpy.ndarray, ... )
        """
        logging.debug("copy_constant_memory_args called")
        logging.debug("current module: " + str(self.current_module))
        for k, v in cmem_args.items():
            symbol = self.current_module.get_global(k)[0]
            logging.debug("copying to symbol: " + str(symbol))
            logging.debug("array to be copied: ")
            logging.debug(v.nbytes)
            logging.debug(v.dtype)
            logging.debug(v.flags)
            drv.memcpy_htod(symbol, v)

    def copy_shared_memory_args(self, smem_args):
        """Add shared memory arguments to the kernel."""
        self.smem_size = smem_args["size"]

    def copy_texture_memory_args(self, texmem_args):
        """Adds texture memory arguments to the most recently compiled module.

        :param texmem_args: A dictionary containing the data to be passed to the
            device texture memory. See tune_kernel().
        :type texmem_args: dict
        """
        filter_mode_map = {
            "point": drv.filter_mode.POINT,
            "linear": drv.filter_mode.LINEAR,
        }
        address_mode_map = {
            "border": drv.address_mode.BORDER,
            "clamp": drv.address_mode.CLAMP,
            "mirror": drv.address_mode.MIRROR,
            "wrap": drv.address_mode.WRAP,
        }

        logging.debug("copy_texture_memory_args called")
        logging.debug("current module: " + str(self.current_module))
        self.texrefs = []
        for k, v in texmem_args.items():
            tex = self.current_module.get_texref(k)
            self.texrefs.append(tex)

            logging.debug("copying to texture: " + str(k))
            if not isinstance(v, dict):
                data = v
            else:
                data = v["array"]
            logging.debug("texture to be copied: ")
            logging.debug(data.nbytes)
            logging.debug(data.dtype)
            logging.debug(data.flags)

            drv.matrix_to_texref(data, tex, order="C")

            if isinstance(v, dict):
                if "address_mode" in v and v["address_mode"] is not None:
                    # address_mode is set per axis
                    amode = v["address_mode"]
                    if not isinstance(amode, list):
                        amode = [amode] * data.ndim
                    for i, m in enumerate(amode):
                        try:
                            if m is not None:
                                tex.set_address_mode(i, address_mode_map[m])
                        except KeyError:
                            raise ValueError("Unknown address mode: " + m)
                if "filter_mode" in v and v["filter_mode"] is not None:
                    fmode = v["filter_mode"]
                    try:
                        tex.set_filter_mode(filter_mode_map[fmode])
                    except KeyError:
                        raise ValueError("Unknown filter mode: " + fmode)
                if "normalized_coordinates" in v and v["normalized_coordinates"]:
                    tex.set_flags(tex.get_flags() | drv.TRSF_NORMALIZED_COORDINATES)

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        """Runs the CUDA kernel passed as 'func'.

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
        if stream is None:
            stream = self.stream
        func(*gpu_args, block=threads, grid=grid, stream=stream, shared=self.smem_size, texrefs=self.texrefs)

    def memset(self, allocation, value, size):
        """Set the memory in allocation to the value in value.

        :param allocation: A GPU memory allocation unit
        :type allocation: pycuda.driver.DeviceAllocation

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        drv.memset_d8(allocation, value, size)

    def memcpy_dtoh(self, dest, src):
        """Perform a device to host memory copy.

        :param dest: A numpy array (or CPU PyTorch tensor) in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: pycuda.driver.DeviceAllocation
        """
        drv.memcpy_dtoh(np.asarray(dest), src)

    def memcpy_htod(self, dest, src):
        """Perform a host to device memory copy.

        :param dest: A GPU memory allocation unit
        :type dest: pycuda.driver.DeviceAllocation

        :param src: A numpy array (or CPU PyTorch tensor) in host memory, or a device array
            that implements the CUDA Array Interface
        :type src: numpy.ndarray
        """
        device_array = get_device_array(src)
        if device_array is not None:
            drv.memcpy_dtod(dest, *device_array)
        else:
            drv.memcpy_htod(dest, np.asarray(src))

    units = {"time": "ms", "power": "s,mW", "energy": "J"}

    last_selected_device = None
    last_selected_context = None
