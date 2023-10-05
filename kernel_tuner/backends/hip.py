"""This module contains all HIP specific kernel_tuner functions."""

import ctypes
import ctypes.util
import logging

import numpy as np

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.hip import HipRuntimeObserver

try:
    from pyhip import hip, hiprtc
except ImportError:
    hip = None
    hiprtc = None

dtype_map = {
    "bool": ctypes.c_bool,
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
}

hipSuccess = 0

class HipFunctions(GPUBackend):
    """Class that groups the HIP functions on maintains state about the device."""

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """Instantiate HipFunctions object used for interacting with the HIP device.

        Instantiating this object will inspect and store certain device properties at
        runtime, which are used during compilation and/or execution of kernels by the
        kernel tuner. It also maintains a reference to the most recently compiled
        source module for copying data to constant memory before kernel launch.

        :param device: Number of HIP device to use for this context
        :type device: int

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int
        """
        if not hip or not hiprtc:
            raise ImportError("Unable to import PyHIP, make sure PYTHONPATH includes PyHIP, or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#hip-and-pyhip.")

        # embedded in try block to be able to generate documentation
        # and run tests without pyhip installed
        logging.debug("HipFunction instantiated")

        self.hipProps = hip.hipGetDeviceProperties(device)

        self.name = self.hipProps._name.decode('utf-8')
        self.max_threads = self.hipProps.maxThreadsPerBlock
        self.device = device
        self.compiler_options = compiler_options or []
        self.iterations = iterations

        env = dict()
        env["device_name"] = self.name
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        self.env = env

        # create a stream and events
        self.stream = hip.hipStreamCreate()
        self.start = hip.hipEventCreate()
        self.end = hip.hipEventCreate()

        # default dynamically allocated shared memory size, can be overwritten using smem_args
        self.smem_size = 0

        self.current_module = None

        # setup observers
        self.observers = observers or []
        self.observers.append(HipRuntimeObserver(self))
        for obs in self.observers:
            obs.register_device(self)


    def ready_argument_list(self, arguments):
        """Ready argument list to be passed to the HIP function.

        :param arguments: List of arguments to be passed to the HIP function.
            The order should match the argument list on the HIP function.
            Allowed values are np.ndarray, and/or np.int32, np.float32, and so on.
        :type arguments: list(numpy objects)

        :returns: Ctypes structure of arguments to be passed to the HIP function.
        :rtype: ctypes structure
        """
        logging.debug("HipFunction ready_argument_list called")

        ctype_args = []
        data_ctypes = None
        for arg in arguments:
            dtype_str = str(arg.dtype)
            # Allocate space on device for array and convert to ctypes
            if isinstance(arg, np.ndarray):
                if dtype_str in dtype_map.keys():
                    device_ptr = hip.hipMalloc(arg.nbytes)
                    data_ctypes = arg.ctypes.data_as(ctypes.POINTER(dtype_map[dtype_str]))
                    hip.hipMemcpy_htod(device_ptr, data_ctypes, arg.nbytes)
                    # may be part of run_kernel, return allocations here instead
                    ctype_args.append(device_ptr)
                else:
                    raise TypeError("unknown dtype for ndarray")
            # Convert valid non-array arguments to ctypes
            elif isinstance(arg, np.generic):
                data_ctypes = dtype_map[dtype_str](arg)
                ctype_args.append(data_ctypes)
            else:
                raise ValueError(f"Invalid argument type {type(arg)}, {arg}")

        return ctype_args


    def compile(self, kernel_instance):
        """Call the HIP compiler to compile the kernel, return the function.

        :param kernel_instance: An object representing the specific instance of the tunable kernel
            in the parameter space.
        :type kernel_instance: kernel_tuner.core.KernelInstance

        :returns: An ctypes function that can be called directly.
        :rtype: ctypes._FuncPtr
        """
        logging.debug("HipFunction compile called")

        #Format and create program
        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name
        if 'extern "C"' not in kernel_string:
            kernel_string = 'extern "C" {\n' + kernel_string + "\n}"
        kernel_ptr = hiprtc.hiprtcCreateProgram(kernel_string, kernel_name, [], [])

        try:
            #Compile based on device (Not yet tested for non-AMD devices)
            plat = hip.hipGetPlatformName()
            if plat == "amd":
                options_list = [f'--offload-arch={self.hipProps.gcnArchName}']
                options_list.extend(self.compiler_options)
                hiprtc.hiprtcCompileProgram(kernel_ptr, options_list)
            else:
                options_list = []
                options_list.extend(self.compiler_options)
                hiprtc.hiprtcCompileProgram(kernel_ptr, options_list)

            #Get module and kernel from compiled kernel string
            code = hiprtc.hiprtcGetCode(kernel_ptr)
            module = hip.hipModuleLoadData(code)
            self.current_module = module
            kernel = hip.hipModuleGetFunction(module, kernel_name)

        except Exception as e:
            log = hiprtc.hiprtcGetProgramLog(kernel_ptr)
            print(log)
            raise e

        return kernel

    def start_event(self):
        """Records the event that marks the start of a measurement."""
        logging.debug("HipFunction start_event called")

        hip.hipEventRecord(self.start, self.stream)

    def stop_event(self):
        """Records the event that marks the end of a measurement."""
        logging.debug("HipFunction stop_event called")

        hip.hipEventRecord(self.end, self.stream)

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        logging.debug("HipFunction kernel_finished called")

        # Query the status of the event
        return hip.hipEventQuery(self.end)

    def synchronize(self):
        """Halts execution until device has finished its tasks."""
        logging.debug("HipFunction synchronize called")

        hip.hipDeviceSynchronize()

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        """Runs the HIP kernel passed as 'func'.

        :param func: A HIP kernel compiled for this specific kernel configuration
        :type func: ctypes pionter

        :param gpu_args: A ctypes structure of arguments to the kernel, order should match the
            order in the code. Allowed values are either variables in global memory
            or single values passed by value.
        :type gpu_args: ctypes structure

        :param threads: A tuple listing the number of threads in each dimension of
            the thread block
        :type threads: tuple(int, int, int)

        :param grid: A tuple listing the number of thread blocks in each dimension
            of the grid
        :type grid: tuple(int, int, int)
        """
        logging.debug("HipFunction run_kernel called")

        if stream is None:
            stream = self.stream

        # Determine the types of the fields in the structure
        field_types = [type(x) for x in gpu_args]

        # Define a new ctypes structure with the inferred layout
        class ArgListStructure(ctypes.Structure):
            _fields_ = [(f'field{i}', t) for i, t in enumerate(field_types)]
            def __getitem__(self, key):
                return getattr(self, self._fields_[key][0])

        ctype_args = ArgListStructure(*gpu_args)

        hip.hipModuleLaunchKernel(func,
                                  grid[0], grid[1], grid[2],
                                  threads[0], threads[1], threads[2],
                                  self.smem_size,
                                  stream,
                                  ctype_args)

    def memset(self, allocation, value, size):
        """Set the memory in allocation to the value in value.

        :param allocation: A GPU memory allocation unit
        :type allocation: ctypes ptr

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        logging.debug("HipFunction memset called")

        hip.hipMemset(allocation, value, size)

    def memcpy_dtoh(self, dest, src):
        """Perform a device to host memory copy.

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: ctypes ptr
        """
        logging.debug("HipFunction memcpy_dtoh called")

        # Format arguments to correct type and perform memory copy
        dtype_str = str(dest.dtype)
        dest_c = dest.ctypes.data_as(ctypes.POINTER(dtype_map[dtype_str]))
        hip.hipMemcpy_dtoh(dest_c, src, dest.nbytes)

    def memcpy_htod(self, dest, src):
        """Perform a host to device memory copy.

        :param dest: A GPU memory allocation unit
        :type dest: ctypes ptr

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        logging.debug("HipFunction memcpy_htod called")

        # Format arguments to correct type and perform memory copy
        dtype_str = str(src.dtype)
        src_c = src.ctypes.data_as(ctypes.POINTER(dtype_map[dtype_str]))
        hip.hipMemcpy_htod(dest, src_c, src.nbytes)

    def copy_constant_memory_args(self, cmem_args):
        """Adds constant memory arguments to the most recently compiled module.

        :param cmem_args: A dictionary containing the data to be passed to the
            device constant memory. The format to be used is as follows: A
            string key is used to name the constant memory symbol to which the
            value needs to be copied. Similar to regular arguments, these need
            to be numpy objects, such as numpy.ndarray or numpy.int32, and so on.
        :type cmem_args: dict( string: numpy.ndarray, ... )
        """
        logging.debug("HipFunction copy_constant_memory_args called")

        # Iterate over dictionary
        for k, v in cmem_args.items():
            #Get symbol pointer
            symbol_ptr, _ = hip.hipModuleGetGlobal(self.current_module, k)

            #Format arguments and perform memory copy
            dtype_str = str(v.dtype)
            v_c = v.ctypes.data_as(ctypes.POINTER(dtype_map[dtype_str]))
            hip.hipMemcpy_htod(symbol_ptr, v_c, v.nbytes)

    def copy_shared_memory_args(self, smem_args):
        """Add shared memory arguments to the kernel."""
        logging.debug("HipFunction copy_shared_memory_args called")

        self.smem_size = smem_args["size"]

    def copy_texture_memory_args(self, texmem_args):
        """Copy texture memory arguments. Not yet implemented."""
        logging.debug("HipFunction copy_texture_memory_args called")

        raise NotImplementedError("HIP backend does not support texture memory")

    units = {"time": "ms", "power": "s,mW", "energy": "J"}
