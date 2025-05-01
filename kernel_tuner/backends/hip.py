"""This module contains all HIP specific kernel_tuner functions."""

import ctypes
import ctypes.util
import logging

import numpy as np

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.hip import HipRuntimeObserver

try:
    from hip import hip, hiprtc
except (ImportError, RuntimeError):
    hip = None
    hiprtc = None

dtype_map = {
    "bool": ctypes.c_bool,
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "float16": ctypes.c_int16,
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


def hip_check(call_result):
    """helper function to check return values of hip calls"""
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


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
            raise ImportError(
                "Unable to import HIP Python, check https://kerneltuner.github.io/kernel_tuner/stable/install.html#hip-and-hip-python."
            )

        # embedded in try block to be able to generate documentation
        # and run tests without HIP Python installed
        logging.debug("HipFunction instantiated")

        # Get device properties
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, device))

        self.name = props.name.decode("utf-8")
        self.max_threads = props.maxThreadsPerBlock
        self.device = device
        self.compiler_options = compiler_options or []
        self.iterations = iterations

        env = dict()
        env["device_name"] = self.name
        env["iterations"] = self.iterations
        env["compiler_options"] = compiler_options
        self.env = env

        # Create stream and events
        self.stream = hip_check(hip.hipStreamCreate())
        self.start = hip_check(hip.hipEventCreate())
        self.end = hip_check(hip.hipEventCreate())

        # Default dynamically allocated shared memory size
        self.smem_size = 0

        self.current_module = None

        # Setup observers
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
        :returns: List of arguments to be passed to the HIP function.
        :rtype: list
        """
        logging.debug("HipFunction ready_argument_list called")
        prepared_args = []

        for arg in arguments:
            dtype_str = str(arg.dtype)

            # Handle numpy arrays
            if isinstance(arg, np.ndarray):
                if dtype_str in dtype_map.keys():
                    # Allocate device memory
                    device_ptr = hip_check(hip.hipMalloc(arg.nbytes))

                    # Copy data to device using hipMemcpy
                    hip_check(hip.hipMemcpy(device_ptr, arg, arg.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

                    prepared_args.append(device_ptr)
                else:
                    raise TypeError(f"Unknown dtype {dtype_str} for ndarray")

            # Handle numpy scalar types
            elif isinstance(arg, np.generic):
                # Convert numpy scalar to corresponding ctypes
                ctype_arg = dtype_map[dtype_str](arg)
                prepared_args.append(ctype_arg)

            else:
                raise ValueError(f"Invalid argument type {type(arg)}, {arg}")

        return prepared_args

    def compile(self, kernel_instance):
        """Call the HIP compiler to compile the kernel, return the function.

        :param kernel_instance: An object representing the specific instance of the tunable kernel
            in the parameter space.
        :type kernel_instance: kernel_tuner.core.KernelInstance

        :returns: A HIP kernel function that can be called.
        :rtype: hipFunction_t
        """
        logging.debug("HipFunction compile called")

        # Format kernel string
        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name
        if 'extern "C"' not in kernel_string:
            kernel_string = 'extern "C" {\n' + kernel_string + "\n}"

        # Create program
        prog = hip_check(hiprtc.hiprtcCreateProgram(kernel_string.encode(), kernel_name.encode(), 0, [], []))

        try:
            # Get device properties
            props = hip.hipDeviceProp_t()
            hip_check(hip.hipGetDeviceProperties(props, 0))

            # Setup compilation options
            arch = props.gcnArchName
            cflags = [b"--offload-arch=" + arch]
            cflags.extend([opt.encode() if isinstance(opt, str) else opt for opt in self.compiler_options])

            # Compile program
            (err,) = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
            if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
                # Get compilation log if there's an error
                log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
                log = bytearray(log_size)
                hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
                raise RuntimeError(log.decode())

            # Get compiled code
            code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
            code = bytearray(code_size)
            hip_check(hiprtc.hiprtcGetCode(prog, code))

            # Load module and get function
            module = hip_check(hip.hipModuleLoadData(code))
            self.current_module = module
            kernel = hip_check(hip.hipModuleGetFunction(module, kernel_name.encode()))

        except Exception as e:
            # Cleanup
            hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))
            raise e

        return kernel

    def start_event(self):
        """Records the event that marks the start of a measurement."""
        logging.debug("HipFunction start_event called")

        hip_check(hip.hipEventRecord(self.start, self.stream))

    def stop_event(self):
        """Records the event that marks the end of a measurement."""
        logging.debug("HipFunction stop_event called")

        hip_check(hip.hipEventRecord(self.end, self.stream))

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        logging.debug("HipFunction kernel_finished called")

        # ROCm HIP returns (hipError_t, bool) for hipEventQuery
        status = hip.hipEventQuery(self.end)
        if status[0] == hip.hipError_t.hipSuccess:
            return True
        elif status[0] == hip.hipError_t.hipErrorNotReady:
            return False
        else:
            hip_check(status)

    def synchronize(self):
        """Halts execution until device has finished its tasks."""
        logging.debug("HipFunction synchronize called")

        hip_check(hip.hipDeviceSynchronize())

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        """Runs the HIP kernel passed as 'func'.

        :param func: A HIP kernel compiled for this specific kernel configuration
        :type func: hipFunction_t

        :param gpu_args: List of arguments to pass to the kernel. Can be DeviceArray
                        objects or ctypes values
        :type gpu_args: list

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

        # Create dim3 objects for grid and block dimensions
        grid_dim = hip.dim3(x=grid[0], y=grid[1], z=grid[2])
        block_dim = hip.dim3(x=threads[0], y=threads[1], z=threads[2])

        # Launch kernel with the arguments
        hip_check(
            hip.hipModuleLaunchKernel(
                func,
                *grid_dim,
                *block_dim,
                sharedMemBytes=self.smem_size,
                stream=stream,
                kernelParams=None,
                extra=tuple(gpu_args),
            )
        )

    def memset(self, allocation, value, size):
        """Set the memory in allocation to the value in value.

        :param allocation: A GPU memory allocation (DeviceArray)
        :type allocation: DeviceArray or int

        :param value: The value to set the memory to
        :type value: int (8-bit unsigned)

        :param size: The size of to the allocation unit in bytes
        :type size: int
        """
        logging.debug("HipFunction memset called")

        hip_check(hip.hipMemset(allocation, value, size))

    def memcpy_dtoh(self, dest, src):
        """Perform a device to host memory copy.

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: DeviceArray or int
        """
        logging.debug("HipFunction memcpy_dtoh called")

        hip_check(hip.hipMemcpy(dest, src, dest.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    def memcpy_htod(self, dest, src):
        """Perform a host to device memory copy.

        :param dest: A GPU memory allocation unit
        :type dest: DeviceArray or int

        :param src: A numpy array in host memory to copy from
        :type src: numpy.ndarray
        """
        logging.debug("HipFunction memcpy_htod called")

        hip_check(hip.hipMemcpy(dest, src, src.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    def copy_constant_memory_args(self, cmem_args):
        """Adds constant memory arguments to the most recently compiled module.

        :param cmem_args: A dictionary containing the data to be passed to the
            device constant memory. The format to be used is as follows: A
            string key is used to name the constant memory symbol to which the
            value needs to be copied. Similar to regular arguments, these need
            to be numpy objects, such as numpy.ndarray or numpy.int32, and so on.
        :type cmem_args: dict(string: numpy.ndarray, ...)
        """
        logging.debug("HipFunction copy_constant_memory_args called")

        # Iterate over dictionary
        for symbol_name, data in cmem_args.items():
            # Get symbol pointer and size using hipModuleGetGlobal
            dptr, _ = hip_check(hip.hipModuleGetGlobal(self.current_module, symbol_name.encode()))

            # Copy data to the global memory location
            hip_check(hip.hipMemcpy(dptr, data, data.nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    def copy_shared_memory_args(self, smem_args):
        """Add shared memory arguments to the kernel."""
        logging.debug("HipFunction copy_shared_memory_args called")

        self.smem_size = smem_args["size"]

    def copy_texture_memory_args(self, texmem_args):
        """Copy texture memory arguments. Not yet implemented."""
        logging.debug("HipFunction copy_texture_memory_args called")

        raise NotImplementedError("HIP backend does not support texture memory")

    units = {"time": "ms", "power": "s,mW", "energy": "J"}
