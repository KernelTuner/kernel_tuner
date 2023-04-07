"""This module contains all HIP specific kernel_tuner functions"""

import numpy as np
import ctypes
import ctypes.util
from collections import namedtuple
import os
import sys

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

    def ready_argument_list(self, arguments):
        """ready argument list to be passed to the HIP function
        :param arguments: List of arguments to be passed to the HIP function.
            The order should match the argument list on the HIP function.
            Allowed values are np.ndarray, and/or np.int32, np.float32, and so on.
        :type arguments: list(numpy objects)
        :returns: A ctypes structure that can be passed to the HIP function.
        :rtype: ctypes.Structure
        """

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
        
        # Determine the number of fields in the structure
        num_fields = len(ctype_args)
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
        self.start = hip.hipEventCreate()
        hip.hipEventRecord(self.start, self.stream)

    def stop_event(self):
        """Records the event that marks the end of a measurement"""
        self.end = hip.hipEventCreate()
        hip.hipEventRecord(self.end, self.stream)

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise"""
        
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
        """This method must implement a barrier that halts execution until device has finished its tasks."""
        pass

    def run_kernel(self, func, gpu_args, threads, grid, stream):
        """This method must implement the execution of the kernel on the device."""
        pass

    def memset(self, allocation, value, size):
        """This method must implement setting the memory to a value on the device."""
        pass

    def memcpy_dtoh(self, dest, src):
        """This method must implement a device to host copy."""
        pass

    def memcpy_htod(self, dest, src):
        """This method must implement a host to device copy."""
        pass

    def copy_constant_memory_args(self, cmem_args):
        """This method must implement the allocation and copy of constant memory to the GPU."""
        pass

    def copy_shared_memory_args(self, smem_args):
        """This method must implement the dynamic allocation of shared memory on the GPU."""
        pass

    def copy_texture_memory_args(self, texmem_args):
        """This method must implement the allocation and copy of texture memory to the GPU."""
        pass

    units = {"time": "ms"}
