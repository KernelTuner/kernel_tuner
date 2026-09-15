"""This module contains all NVIDIA cuda-python specific kernel_tuner functions."""

import numpy as np
import uuid
import os
import subprocess
import tempfile

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.nvcuda import CudaRuntimeObserver
from kernel_tuner.util import SkippableFailure
from kernel_tuner.utils.nvcuda import cuda_error_check, to_valid_nvrtc_gpu_arch_cc, find_cuda_home, _check

def preload_python_nvrtc():
    """Preload the libnvrtc.so library into the process memory to avoid conflicts with Julia's artifacts."""

    import ctypes as ct
    import sysconfig

    # search site-packages for nvidia-cuda-nvrtc wheels
    site_packages = sysconfig.get_paths()["purelib"]
    nvidia_path = os.path.join(site_packages, "nvidia", "nvrtc", "lib")

    # fall back to CUDA_HOME / CUDA_PATH if present
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    cuda_home_lib = os.path.join(cuda_home, "lib64") if cuda_home else None

    # search for libnvrtc.so in the search directories and load it into process memory
    search_dirs = [nvidia_path, cuda_home_lib]
    for d in search_dirs:
        if d and os.path.exists(d):
            for file in os.listdir(d):
                if file.startswith("libnvrtc.so"):
                    full_path = os.path.join(d, file)
                    # force load into process memory before Julia loads its artifacts
                    ct.CDLL(full_path, mode=ct.RTLD_GLOBAL)
                    return True
    return False

# embedded in try block to be able to generate documentation
# and run tests without cuda-python installed
try:
    from cuda.bindings import driver, nvrtc, runtime
    preload_python_nvrtc()
except ImportError:
    try:
        # backward compatibility hack for older cuda-python versions
        from cuda import cuda as driver
        from cuda import cudart as runtime
        from cuda import nvrtc as nvrtc
    except ImportError:
        driver = None

try:
    from cuda.core import Kernel as CudaCoreKernel, launch as cuda_core_launch, LaunchConfig as CudaCoreConfig
    _cuda_core_available = True
except ImportError:
    _cuda_core_available = False


class CudaFunctions(GPUBackend):
    """Class that groups the Cuda functions and it maintains state about the device."""

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """Instantiate CudaFunctions object used for interacting with the CUDA device.

        Instantiating this object will inspect and store certain device properties at
        runtime, which are used during compilation and/or execution of kernels by the
        kernel tuner. It also maintains a reference to the most recently compiled
        source module for copying data to constant memory before kernel launch.

        :param device: Number of CUDA device to use for this context
        :type device: int

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int

        :param compiler_options: Compiler options for the CUDA runtime compiler

        :param observers: List of Observer type objects
        """
        self.allocations = []
        self.texrefs = []
        if not driver:
            raise ImportError(
                "cuda-python not installed, install using 'pip install cuda-python', or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda."
            )

        # initialize and select device
        err = driver.cuInit(0)
        cuda_error_check(err)
        err, self.device = driver.cuDeviceGet(device)
        cuda_error_check(err)
        err, self.context = driver.cuDevicePrimaryCtxRetain(device)
        cuda_error_check(err)
        if CudaFunctions.last_selected_device != device:
            err = driver.cuCtxSetCurrent(self.context)
            cuda_error_check(err)
            CudaFunctions.last_selected_device = device

        # compute capabilities and device properties
        err, major = runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device)
        cuda_error_check(err)
        err, minor = runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device)
        cuda_error_check(err)
        err, self.max_threads = runtime.cudaDeviceGetAttribute(
            runtime.cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock, device
        )
        cuda_error_check(err)
        self.cc = f"{major}{minor}"
        self.iterations = iterations
        self.current_module = None
        self.current_library = None
        self.func = None
        self.compiler_options = compiler_options or []

        # create a stream and events
        err, self.stream = driver.cuStreamCreate(0)
        cuda_error_check(err)
        err, self.start = driver.cuEventCreate(0)
        cuda_error_check(err)
        err, self.end = driver.cuEventCreate(0)
        cuda_error_check(err)
        self.current_sm_percentage = 100
        self.green_ctx_cache = {}
        self.green_ctx = None

        # default dynamically allocated shared memory size, can be overwritten using smem_args
        self.smem_size = 0

        # collect environment information
        err, device_properties = runtime.cudaGetDeviceProperties(device)
        cuda_error_check(err)
        env = dict()
        env["uuid"] = str(uuid.UUID(bytes=device_properties.uuid.bytes))
        env["device_name"] = device_properties.name.decode()
        env["cuda_version"] = driver.CUDA_VERSION
        env["compute_capability"] = self.cc
        env["iterations"] = self.iterations
        env["compiler_options"] = self.compiler_options
        env["device_properties"] = str(device_properties).replace("\n", ", ")

        # We must use `cudaDeviceGetPCIBusId` to get the PCI bus string
        # It returns a series of bytes containing a null byte, not a `str`
        err, pci_bus = runtime.cudaDeviceGetPCIBusId(32, device) # 32 = length?
        cuda_error_check(err)
        env["pci_bus_id"] = pci_bus.decode("ascii").split("\x00", 1)[0]

        self.env = env
        self.name = env["device_name"]

        # setup observers
        self.observers = observers or []
        self.observers.append(CudaRuntimeObserver(self))
        for observer in self.observers:
            observer.register_device(self)

    def __del__(self):
        # Cleanup streams and green contexts, if any
        if self.green_ctx_cache:
            for val in self.green_ctx_cache.values():
                green_ctx, stream, _ = val
                _check(driver.cuStreamDestroy(stream))
                _check(driver.cuGreenCtxDestroy(green_ctx))

        # Cleanup
        for device_memory in self.allocations:
            if isinstance(device_memory, driver.CUdeviceptr):
                _check(driver.cuMemFree(device_memory))


    def set_sm_percentage(self, sm_percentage):
        """ Set the active SM percentage

        Create a CUDA green context owning ~`sm_percentage` of the device's SMs
        and a stream bound to it. Kernels launched afterwards are restricted
        to that SM partition. Green contexts are cached in self.green_ctx_cache.
        The actual number of SMs in the partition may not exactly match the
        requested percentage. An observer may be used to query:

         *   Currently assigned number of SMs: self.assigned_sm_count
         *   Currently requested SM percentage: self.current_sm_percentage

        Requires: CUDA >= 12.4 and a GPU that supports SM partitioning.
        """

        if not 0 < sm_percentage <= 100:
            raise ValueError("sm_percentage must be in (0, 100]")

        # Check if sm_percentage is already applied
        if sm_percentage == self.current_sm_percentage:
            return

        # Check if this sm_percentage has been requested before
        if sm_percentage in self.green_ctx_cache:
            self.green_ctx, self.stream, self.assigned_sm_count = self.green_ctx_cache[sm_percentage]
            self.current_sm_percentage = sm_percentage
            return

        # Get total SMs and desired percentage
        total_sms = _check(driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self.device))
        want = max(1, round(total_sms * sm_percentage / 100.0))

        # Full SM resource pool of the device.
        sm_resource = _check(driver.cuDeviceGetDevResource(
            self.device, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))

        # Split off one group of at least `want` SMs. The driver rounds up to the
        # device's partitioning granularity, so the actual count may be larger.
        groups, _nb, _remaining = _check(driver.cuDevSmResourceSplitByCount(
            1,            # number of groups requested
            sm_resource,  # input resource
            0,            # useFlags (0 = default)
            want,         # minCount of SMs per group
        ))
        group = groups[0]
        assigned = group.sm.smCount

        # Descriptor -> green context.
        desc = _check(driver.cuDevResourceGenerateDesc([group], 1))
        green_ctx = _check(driver.cuGreenCtxCreate(
            desc, self.device, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))

        # A stream from the green context confines launches to its SMs.
        stream = _check(driver.cuGreenCtxStreamCreate(
            green_ctx,
            driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
            0,  # priority
        ))
        self.green_ctx_cache[sm_percentage] = (green_ctx, stream, assigned)
        self.green_ctx = green_ctx
        self.stream = stream
        self.assigned_sm_count = assigned
        self.current_sm_percentage = sm_percentage


    def ready_argument_list(self, arguments):
        """Ready argument list to be passed to the kernel, allocates gpu mem.

        :param arguments: List of arguments to be passed to the kernel.
            The order should match the argument list on the CUDA kernel.
            Allowed values are numpy.ndarray, and/or numpy.int32, numpy.float32, and so on.
        :type arguments: list(numpy objects)

        :returns: A list of arguments that can be passed to an CUDA kernel.
        :rtype: list( pycuda.driver.DeviceAllocation, numpy.int32, ... )
        """
        gpu_args = []
        for arg in arguments:
            # if arg is a numpy array copy it to device
            if isinstance(arg, np.ndarray):
                err, device_memory = driver.cuMemAlloc(arg.nbytes)
                cuda_error_check(err)
                self.allocations.append(device_memory)
                gpu_args.append(device_memory)
                self.memcpy_htod(device_memory, arg)
            # if not array, just pass along
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

        :returns: A kernel that can be launched by the CUDA runtime
        :rtype:
        """
        kernel_string = kernel_instance.kernel_string
        kernel_name = kernel_instance.name
        compiler_options = list(self.compiler_options)

        # Detect Tile kernels: user passes "-enable-tile" (NVRTC flag) as a signal.
        # Tile kernels are compiled with nvcc -tilecubin rather than NVRTC.
        is_tile_kernel = any(str(opt).strip() == "-enable-tile" for opt in compiler_options)

        if is_tile_kernel:
            if not _cuda_core_available:
                raise RuntimeError("Tile kernels require 'cuda-core' (pip install cuda-core)")
            self.func = self._compile_tile_kernel_nvcc(kernel_string, kernel_name, compiler_options)
            self.num_regs = 0
        else:
            expression_name = str.encode(kernel_name)

            # Add -std=c++11
            if not any(opt.startswith(("-std=", "--std=")) for opt in self.compiler_options):
                compiler_options.append("--std=c++11")

            # Add -arch
            if not any(opt.startswith(("-arch", "--arch", "--gpu-architecture=")) for opt in self.compiler_options):
                arch_val = to_valid_nvrtc_gpu_arch_cc(self.cc)
                compiler_options.append(f"--gpu-architecture=compute_{arch_val}")

            # Add CUDA home to include path
            cuda_home = find_cuda_home()
            if cuda_home:
                cuda_include = os.path.join(cuda_home, "include")
                compiler_options.append(f"-I{cuda_include}")

            # nvrtcCompileProgram requires bytes instead of str
            compiler_options = [str(opt).encode("UTF-8") for opt in compiler_options]

            err, program = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b"CUDAProgram", 0, [], [])
            try:
                cuda_error_check(err)
                # Add the kernel as an expression. This is necessary for templated kernels to ensure that the
                # compiler actually instantiates the kernel that we want to compile.
                err = nvrtc.nvrtcAddNameExpression(program, expression_name)
                cuda_error_check(err)

                # Compile the program
                err = nvrtc.nvrtcCompileProgram(program, len(compiler_options), compiler_options)
                cuda_error_check(err)

                # Get the PTX
                err, size = nvrtc.nvrtcGetPTXSize(program)
                cuda_error_check(err)
                buff = b" " * size
                err = nvrtc.nvrtcGetPTX(program, buff)
                cuda_error_check(err)

                # Load the module
                err, self.current_module = _load_module_with_logs(buff)

                # If the compile succeeded but loading the PTX failed it is most likely
                # that the kernel uses too much shared memory
                if err == driver.CUresult.CUDA_ERROR_INVALID_PTX:
                    raise SkippableFailure("uses too much shared data")
                else:
                    cuda_error_check(err)

                # Get the lowered name (resolves C++ mangling) and look up the function
                err, lowered_name = nvrtc.nvrtcGetLoweredName(program, expression_name)
                cuda_error_check(err)
                err, self.func = driver.cuModuleGetFunction(self.current_module, lowered_name)
                if err == driver.CUresult.CUDA_ERROR_NOT_FOUND:
                    err, self.func = driver.cuModuleGetFunction(self.current_module, expression_name)
                cuda_error_check(err)

                # get the number of registers per thread used in this kernel
                num_regs = driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func)
                assert num_regs[0] == 0, f"Retrieving number of registers per thread unsuccesful: code {num_regs[0]}"
                self.num_regs = num_regs[1]

            except RuntimeError as re:
                _, n = nvrtc.nvrtcGetProgramLogSize(program)
                log = b" " * n
                nvrtc.nvrtcGetProgramLog(program, log)
                print(log.decode("utf-8"))
                raise re

        return self.func

    def _compile_tile_kernel_nvcc(self, kernel_string, kernel_name, compiler_options):
        """Compile a CUDA Tile kernel using nvcc -tilecubin and load it via cuda.core.

        Tile kernels cannot be compiled to PTX via NVRTC and loaded as regular
        modules. Instead, nvcc produces a 'tile cubin' that can be loaded by
        cuda.core.ObjectCode.from_cubin and launched with cuda.core.launch.

        Returns a cuda.core.Kernel that must be launched with block=1.
        """
        from cuda.core import ObjectCode

        nvcc = os.environ.get("KERNEL_TUNER_NVCC", "nvcc")

        # Determine arch: translate --gpu-architecture=compute_XX or -arch=compute_XX
        # to -arch=sm_XX (required for cubin output).
        arch_flag = f"-arch=sm_{self.cc}"
        for opt in compiler_options:
            s = str(opt).strip()
            for prefix in ("--gpu-architecture=compute_", "--gpu-architecture=sm_",
                           "-arch=compute_", "-arch=sm_"):
                if s.startswith(prefix):
                    suffix = s[len(prefix):]
                    arch_flag = f"-arch=sm_{suffix}"
                    break

        # Build the nvcc command, forwarding safe options and skipping NVRTC-only ones.
        _skip = {"-enable-tile"}
        _skip_prefixes = ("--gpu-architecture=", "-arch=")
        nvcc_opts = []
        for opt in compiler_options:
            s = str(opt).strip()
            if s in _skip or any(s.startswith(p) for p in _skip_prefixes):
                continue
            nvcc_opts.append(s)

        with tempfile.TemporaryDirectory() as tmpdir:
            cu_file = os.path.join(tmpdir, "kernel.cu")
            cubin_file = os.path.join(tmpdir, "kernel.cubin")

            with open(cu_file, "w") as f:
                f.write(kernel_string)

            cmd = [nvcc, "-tilecubin", "--tile-only", arch_flag] + nvcc_opts + ["-o", cubin_file, cu_file]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"nvcc Tile kernel compilation failed:\n{e.stderr}"
                ) from e

            cubin_bytes = open(cubin_file, "rb").read()

        # Find the mangled kernel name by loading the cubin as a module and
        # enumerating its functions (same approach as TileGym).
        err, mod = driver.cuModuleLoadData(cubin_bytes)
        cuda_error_check(err)
        err, func_count = driver.cuModuleGetFunctionCount(mod)
        cuda_error_check(err)
        err, functions = driver.cuModuleEnumerateFunctions(func_count, mod)
        cuda_error_check(err)
        # Strip template args (e.g. "vector_add_tile<8>" -> "vector_add_tile") before
        # searching mangled names, since template args are encoded in mangled form.
        base_name = kernel_name.split('<')[0].strip()
        mangled_name = None
        for func in functions:
            err2, name_bytes = driver.cuFuncGetName(func)
            if err2 != driver.CUresult.CUDA_SUCCESS:
                continue
            name = name_bytes.decode() if isinstance(name_bytes, bytes) else name_bytes
            if base_name in name:
                mangled_name = name
                break
        if mangled_name is None and func_count > 0:
            _, name_bytes = driver.cuFuncGetName(functions[0])
            mangled_name = name_bytes.decode() if isinstance(name_bytes, bytes) else name_bytes
        driver.cuModuleUnload(mod)
        if mangled_name is None:
            raise RuntimeError(f"Could not find Tile kernel '{kernel_name}' in compiled cubin")

        # Load the cubin via cuda.core and retrieve the Kernel object.
        # Store the ObjectCode as self.current_library to keep it alive for the
        # duration of the kernel's use (the Kernel handle is only valid while the
        # ObjectCode/library is alive).
        self.current_library = ObjectCode.from_cubin(cubin_bytes)
        return self.current_library.get_kernel(mangled_name)

    def start_event(self):
        """Records the event that marks the start of a measurement."""
        err = runtime.cudaEventRecord(self.start, self.stream)
        cuda_error_check(err)

    def stop_event(self):
        """Records the event that marks the end of a measurement."""
        err = runtime.cudaEventRecord(self.end, self.stream)
        cuda_error_check(err)

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        err = runtime.cudaEventQuery(self.end)
        if err[0] == runtime.cudaError_t.cudaSuccess:
            return True
        else:
            return False

    @staticmethod
    def synchronize():
        """Halts execution until device has finished its tasks."""
        err = runtime.cudaDeviceSynchronize()
        cuda_error_check(err)

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
            err, symbol, _ = driver.cuModuleGetGlobal(self.current_module, str.encode(k))
            cuda_error_check(err)
            err = driver.cuMemcpyHtoD(symbol, v, v.nbytes)
            cuda_error_check(err)

    def copy_shared_memory_args(self, smem_args):
        """Add shared memory arguments to the kernel."""
        self.smem_size = smem_args["size"]

    def copy_texture_memory_args(self, texmem_args):
        """Adds texture memory arguments to the most recently compiled module.

        :param texmem_args: A dictionary containing the data to be passed to the
            device texture memory. See tune_kernel().
        :type texmem_args: dict
        """
        raise NotImplementedError("NVIDIA CUDA backend does not support texture memory")

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        """Runs the CUDA kernel passed as 'func'.

        :param func: A CUDA kernel compiled for this specific kernel configuration
        :type func: cuda.CUfunction

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
        arg_types = list()
        for arg in gpu_args:
            if isinstance(arg, driver.CUdeviceptr):
                arg_types.append(None)
            else:
                arg_types.append(np.ctypeslib.as_ctypes_type(arg.dtype))
        kernel_args = (tuple(gpu_args), tuple(arg_types))
        if _cuda_core_available and isinstance(func, CudaCoreKernel):
            # Tile kernels are launched via cuda.core.launch with block=1;
            # the Tile runtime handles the thread-to-tile mapping internally.
            from cuda.core import Stream as CudaCoreStream
            tile_args = []
            for arg in gpu_args:
                if isinstance(arg, driver.CUdeviceptr):
                    tile_args.append(np.uint64(int(arg)))
                else:
                    tile_args.append(arg)
            # Stream.from_handle takes the raw integer CUstream handle value,
            # the same convention as torch_stream.cuda_stream in PyTorch.
            core_stream = CudaCoreStream.from_handle(int(stream))
            config = CudaCoreConfig(grid=(grid[0], grid[1], grid[2]), block=1, shmem_size=self.smem_size)
            cuda_core_launch(core_stream, config, func, *tile_args)
        else:
            err = driver.cuLaunchKernel(
                func,
                grid[0],
                grid[1],
                grid[2],
                threads[0],
                threads[1],
                threads[2],
                self.smem_size,
                stream,
                kernel_args,
                0,
            )
            cuda_error_check(err)

    @staticmethod
    def memset(allocation, value, size):
        """Set the memory in allocation to the value in value.

        :param allocation: A GPU memory allocation unit
        :type allocation: cupy.ndarray

        :param value: The value to set the memory to
        :type value: a single 8-bit unsigned int

        :param size: The size of to the allocation unit in bytes
        :type size: int

        """
        err = runtime.cudaMemset(allocation, value, size)
        cuda_error_check(err)

    @staticmethod
    def memcpy_dtoh(dest, src):
        """Perform a device to host memory copy.

        :param dest: A numpy array in host memory to store the data
        :type dest: numpy.ndarray

        :param src: A GPU memory allocation unit
        :type src: cuda.CUdeviceptr
        """
        err = driver.cuMemcpyDtoH(dest, src, dest.nbytes)
        cuda_error_check(err)

    @staticmethod
    def memcpy_htod(dest, src):
        """Perform a host to device memory copy.

        :param dest: A GPU memory allocation unit
        :type dest: cuda.CUdeviceptr

        :param src: A numpy array in host memory to store the data
        :type src: numpy.ndarray
        """
        err = driver.cuMemcpyHtoD(dest, src, src.nbytes)
        cuda_error_check(err)

    units = {"time": "ms"}

    last_selected_device = None



def _load_module_with_logs(image: bytes, log_size: int = 8192):
    """Load a PTX/cubin/Tile IR image via cuModuleLoadDataEx with
    error/info log capture. Raises RuntimeError with the compiler's
    own diagnostic text on failure, instead of a bare CUresult."""

    error_log = bytearray(log_size)
    info_log = bytearray(log_size)

    options = [
        driver.CUjit_option.CU_JIT_ERROR_LOG_BUFFER,
        driver.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        driver.CUjit_option.CU_JIT_INFO_LOG_BUFFER,
        driver.CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        driver.CUjit_option.CU_JIT_LOG_VERBOSE,
    ]
    option_values = [
        error_log,
        log_size,
        info_log,
        log_size,
        1,  # verbose logging on
    ]

    result = driver.cuModuleLoadDataEx(
        image, len(options), options, option_values
    )

    if result[0] != driver.CUresult.CUDA_SUCCESS:
        err_text = error_log.split(b"\x00", 1)[0].decode(errors="replace")
        info_text = info_log.split(b"\x00", 1)[0].decode(errors="replace")
        raise RuntimeError(
            f"cuModuleLoadDataEx failed: {result[0]}\n"
            f"--- error log ---\n{err_text}\n"
            f"--- info log ---\n{info_text}"
        )

    (module,) = result[1:]
    return result[0], module
