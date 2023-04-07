"""This module contains all HIP specific kernel_tuner functions"""

import numpy as np
import ctypes
import ctypes.util
from collections import namedtuple
import os
import sys
import logging

from kernel_tuner.backends.backend import GPUBackend

try:
    # Load the HIP runtime library
    hip_lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library('hip'))
except ImportError:
    print("Not able to import hip c lib")
    hip_lib = None

# embedded in try block to be able to generate documentation
# and run tests without pyhip installed
PYHIP_PATH = os.environ.get('PYHIP_PATH')  # get the PYHIP_PATH environment variable
try:
    from pyhip import hip, hiprtc
except ImportError:
    print("Not able to import pyhip, check if PYTHONPATH includes PyHIP")
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

class HipFunctions(GPUBackend):
    """Class that groups the HIP functions on maintains state about the device"""

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """instantiate HipFunctions object used for interacting with the HIP device

        Instantiating this object will inspect and store certain device properties at
        runtime, which are used during compilation and/or execution of kernels by the
        kernel tuner. It also maintains a reference to the most recently compiled
        source module for copying data to constant memory before kernel launch.

        :param device: Number of HIP device to use for this context
        :type device: int

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int
        """
        logging.debug("HipFunction instantiated")
        hip.hipInit(0)
        
        hipProps = hip.hipGetDeviceProperties(device)
        self.name = hipProps._name.decode('utf-8')
        self.max_threads = hipProps.maxThreadsPerBlock
        self.device = device
        self.compiler_options = compiler_options

        env = dict()
        self.env = env

        # create a stream and events
        self.stream = hip.hipStreamCreate()
        self.start = hip.hipEventCreate()
        self.end = hip.hipEventCreate()

        self.smem_size = 0

        env["device_name"] = self.name

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the HIP function
        :param arguments: List of arguments to be passed to the HIP function.
            The order should match the argument list on the HIP function.
            Allowed values are np.ndarray, and/or np.int32, np.float32, and so on.
        :type arguments: list(numpy objects)
        :returns: A ctypes structure that can be passed to the HIP function.
        :rtype: ctypes.Structure
        """
        logging.debug("HipFunction ready_argument_list called")
        ctype_args = []
        for i, arg in enumerate(arguments):
            dtype_str = str(arg.dtype)
            if isinstance(arg, np.ndarray):
                if dtype_str in dtype_map.keys():
                    data_ctypes = arg.ctypes.data_as(ctypes.POINTER(dtype_map[dtype_str]))
                else:
                    raise TypeError("unknown dtype for ndarray")        
            elif isinstance(arg, np.generic):
                data_ctypes = dtype_map[dtype_str](arg)
            ctype_args.append(data_ctypes)  
        
        # Determine the types of the fields in the structure
        field_types = [type(x) for x in ctype_args]
        # Define a new ctypes structure with the inferred layout
        class ArgListStructure(ctypes.Structure):
            _fields_ = [(f'field{i}', t) for i, t in enumerate(field_types)]
        ctypes_struct = ArgListStructure()
        # Populate the fields of the structure with values from the list
        for i, value in enumerate(ctype_args):
            setattr(ctypes_struct, f'field{i}', value)
        
        return ctypes_struct
    
    def compile(self, kernel_instance):
        """call the HIP compiler to compile the kernel, return the function
        
        :param kernel_instance: An object representing the specific instance of the tunable kernel
            in the parameter space.
        :type kernel_instance: kernel_tuner.core.KernelInstance
        
        :returns: An ctypes function that can be called directly.
        :rtype: ctypes._FuncPtr
        """
        logging.debug("HipFunction compile called")
        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name

        # if filename is known, use that one
        suffix = kernel_instance.kernel_source.get_user_suffix()

        if suffix is None:
            # select right suffix based on compiler
            suffix = ".cc"

        if ".c" in suffix and 'extern "C"' not in kernel_string:
            kernel_string = 'extern "C" {\n' + kernel_string + "\n}"

        kernel_ptr = hiprtc.hiprtcCreateProgram(kernel_string, kernel_name, [], [])
        
        device_properties = hip.hipGetDeviceProperties(self.device)
        plat = hip.hipGetPlatformName()
        #Compile based on device
        if plat == "amd":
            hiprtc.hiprtcCompileProgram(
                kernel_ptr, [f'--offload-arch={device_properties.gcnArchName}'])
        else:
            hiprtc.hiprtcCompileProgram(kernel_ptr, [])
        code = hiprtc.hiprtcGetCode(kernel_ptr)
        module = hip.hipModuleLoadData(code)
        kernel = hip.hipModuleGetFunction(module, kernel_name)
        
        return kernel
    
    def start_event(self):
        """Records the event that marks the start of a measurement"""
        logging.debug("HipFunction start_event called")
        hip.hipEventRecord(self.start, self.stream)

    def stop_event(self):
        """Records the event that marks the end of a measurement"""
        logging.debug("HipFunction stop_event called")
        hip.hipEventRecord(self.end, self.stream)

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise"""
        logging.debug("HipFunction kernel_finished called")
        
        # Define the argument and return types for hipEventQuery()
        hip_lib.hipEventQuery.argtypes = [ctypes.c_void_p]
        hip_lib.hipEventQuery.restype = ctypes.c_int

        # Query the status of the event
        status = hip_lib.hipEventQuery(self.end)
        if status == hip_lib.hipSuccess:
            # Kernel has finished
            return True
        elif status == hip_lib.hipErrorNotReady:
            # Kernel is still running
            return False
        else:
            # Error occurred
            return False

    def synchronize(self):
        """Halts execution until device has finished its tasks"""
        logging.debug("HipFunction synchronize called")
        hip.hipEventSynchronize(self.end)
        pass

    def run_kernel(self, func, gpu_args, threads, grid, stream):
        """runs the HIP kernel passed as 'func'

        :param func: A PyHIP kernel compiled for this specific kernel configuration
        :type func: ctypes pionter

        :param gpu_args: A list of arguments to the kernel, order should match the
            order in the code. Allowed values are either variables in global memory
            or single values passed by value.
        :type gpu_args: ctypes.Structure

        :param threads: A tuple listing the number of threads in each dimension of
            the thread block
        :type threads: tuple(int, int, int)

        :param grid: A tuple listing the number of thread blocks in each dimension
            of the grid
        :type grid: tuple(int, int)
        """
        logging.debug("HipFunction run_kernel called")
        hip.hipModuleLaunchKernel(func, 
                                  grid[0], grid[1], grid[2], grid[3], 
                                  threads[0], threads[1], threads[2],
                                  self.smem_size,
                                  stream,
                                  gpu_args)
        pass

    def memset(self, allocation, value, size):
        """set the memory in allocation to the value in value

        :param allocation: An Argument for some memory allocation unit
        :type allocation: ctypes ptr

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int
        """
        logging.debug("HipFunction memset called")
        allocation.contents.value = value

    def memcpy_dtoh(self, dest, src):
        """perform a device to host memory copy

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: ctypes ptr
        """
        logging.debug("HipFunction memcpy_dtoh called")
        dtype_str = str(src.dtype)
        hip.hipMemcpy_dtoh(ctypes.byref(dest.ctypes), src, ctypes.sizeof(dtype_map[dtype_str]) * src.size)

    def memcpy_htod(self, dest, src):
        """perform a host to device memory copy

        :param dest: A GPU memory allocation unit
        :type dest: ctypes ptr

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        logging.debug("HipFunction memcpy_htod called")
        dtype_str = str(src.dtype)
        hip.hipMemcpy_htod(dest, ctypes.byref(src.ctypes), ctypes.sizeof(dtype_map[dtype_str]) * src.size)

    def copy_constant_memory_args(self, cmem_args):
        """This method must implement the allocation and copy of constant memory to the GPU."""
        logging.debug("HipFunction copy_constant_memory_args called")

    def copy_shared_memory_args(self, smem_args):
        """This method must implement the dynamic allocation of shared memory on the GPU."""
        logging.debug("HipFunction copy_shared_memory_args called")

    def copy_texture_memory_args(self, texmem_args):
        """This method must implement the allocation and copy of texture memory to the GPU."""
        logging.debug("HipFunction copy_texture_memory_args called")

    units = {"time": "ms"}
