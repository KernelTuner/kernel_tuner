import logging
import inspect
import re
import builtins
import numpy as np

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.generic_python import GenericPythonRuntimeObserver

try:
    import torch
except ImportError:
    torch = None


class GenericPythonFunctions(GPUBackend):
    """Class that groups the Python DSL functions on maintains state about the device."""

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """Instantiate GenericPythonFunctions object used for interacting with the device.
        Currently, only CUDA devices are supported.

        Instantiating this object will inspect and store certain device properties at
        runtime, which are used during compilation and/or execution of kernels by the
        kernel tuner. Compiler options are not supported for GenericPython.

        :param device: Number of CUDA device to use for this context
        :type device: int

        :param iterations: Number of iterations used while benchmarking a kernel, 7 by default.
        :type iterations: int
        """

        if not torch:
            logging.error("Unable to import Torch")
            raise ImportError("Unable to import Torch")

        self.device_id = torch.cuda.current_device()
        self.device_properties = torch.cuda.get_device_properties(self.device_id)
        self.name = torch.cuda.get_device_name(self.device_id)
        self.max_threads = 10**18 # 'inf' to support tile based programming models, which can use less threads then the size of a tile.

        env = {}
        env["device_name"] = self.name
        env["max_threads"] = self.max_threads
        env["iterations"] = iterations
        env["compiler_options"] = compiler_options
        self.env = env

        self.stream = torch.cuda.default_stream()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        # setup observers
        self.observers = observers or []
        self.observers.append(GenericPythonRuntimeObserver(self))
        for obs in self.observers:
            obs.register_device(self)

        self.units = {"time": "ms", "power": "s,mW", "energy": "J"}

        # Variables to be filled in at compile time, needed for running the kernel after compilation.
        self.call_function = None
        self.signature = None
        self.gpu_kwargs = None

        super().__init__(device=device, iterations=iterations, compiler_options=compiler_options, observers=observers)

    def ready_argument_list(self, arguments):
        """Ready argument list to be passed to the kernel. Converts arguments to Torch GPU Tensors or Python
        Scalars. Arguments of built-in Python types are left untouched.

        :param arguments: List of arguments to be passed to the kernel.
            The order should match the argument list on the kernel.
            Allowed values are:
            - numpy.ndarray, and/or numpy.int32, numpy.float32, and so on.
            - Torch Tensors
            - Built-in Python types.
        :type arguments: list

        :returns: A list of arguments that can be passed to the kernel.
        :rtype: list( Torch.Tensor on GPU, int, float, bool, etc. )
        """
        torch_args = []

        for arg in arguments:
            if isinstance(arg, torch.Tensor):
                if arg.dim() == 0: # Scalar tensor, convert to Python scalar
                    torch_args.append(arg.item())
                else:
                    if arg.is_cuda: # already on GPU, need deep copy to not overwrite
                        torch_args.append(arg.clone())
                    else: # Copy from CPU to GPU
                        torch_args.append(arg.contiguous().to("cuda"))
            elif isinstance(arg, np.ndarray): # Convert Numpy CPU array to Torch GPU Tensor
                torch_args.append(torch.from_numpy(arg).to("cuda"))
            elif isinstance(arg, np.generic): # Numpy scalar, convert to Python scalar
                torch_args.append(arg.item())
            elif isinstance(arg, (int, float, bool, str)): # Is already Python
                torch_args.append(arg)
            elif type(arg) in vars(builtins).values():
                torch_args.append(arg)
            else:
                raise TypeError("Unknown argument type: ", type(arg), ". Accepted types are Torch tenors, NumPy arrays and scalars and built-in Python types.")

        return torch_args


    def compile(self, kernel_instance, gpu_args=None):
        """Compile the kernel by executing it once. This enforces that the kernel is cached. 

        :param kernel_instance: The kernel instance containing information such as the kernel_source,
        grid, threads, params, etc.
        :type kernel_instance: KernelInstance

        :param gpu_args: arguments to be passed to the kernel.
        :type gpu_args: list(any)

        :returns: A kernel that can be called with the user-defined call function.
        :rtype: callable
        """
        if kernel_instance.kernel_fn is None:
            raise ValueError("kernel_fn is None, currently Generic Python only supports callable kernel_source")

        if gpu_args is None:
            raise ValueError("gpu_args is None, Generic Python needs gpu args to compile the kernel")

        # The first time we compile, we also set the call function and the signature
        # We need this later to run the kernel.
        if self.call_function is None or self.signature is None:
            self.call_function = kernel_instance.kernel_source.call_function
            self.signature = kernel_instance.kernel_source.signature

        grid = kernel_instance.grid
        threads = kernel_instance.threads
        params = kernel_instance.params

        # If the kernel source is a class, we use the __call__ function as the kernel_function.
        if inspect.isclass(kernel_instance.kernel_fn):
            kernel_function = kernel_instance.kernel_fn()
        elif callable(kernel_instance.kernel_fn):
            kernel_function = kernel_instance.kernel_fn
        else:
            raise TypeError("kernel function is not a class or function")

        # Tuning params can contain kernel arguments. In such cases, create keyword arguments with
        # the values of the tuning params.
        self.gpu_kwargs = {}
        if params is not None:
            for arg_name in self.signature:
                if arg_name in params:
                    self.gpu_kwargs[arg_name] = params[arg_name]

        # Call the user-defined call function in order to compile the kernel.
        self.synchronize()
        self.call_function(kernel_function, gpu_args, self.gpu_kwargs, grid, threads, params)
        self.synchronize()

        return kernel_function

    def start_event(self):
        """Records the event that marks the start of a measurement."""
        self.start.record()

    def stop_event(self):
        """Records the event that marks the end of a measurement."""
        self.end.record()

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        return self.end.query()

    def run_kernel(self, func, gpu_args, threads, grid, params=None):
        """Runs the Python kernel passed as 'func'.

        :param func: A cached Python kernel for this specific kernel configuration
        :type func: callable

        :param gpu_args: A list of arguments to the kernel, order should match the
            order in the code. 
        :type gpu_args: list(any)

        :param threads: A tuple listing the number of threads in each dimension of
            the thread block
        :type threads: tuple(int, int, int)

        :param grid: A tuple listing the number of thread blocks in each dimension
            of the grid
        :type grid: tuple(int, int, int)

        :param params: A dictionary with the tuning params for this specific kernel
            configuration
        :type params: dict
        """
        self.call_function(func, gpu_args, self.gpu_kwargs, grid, threads, params)


    def synchronize(self):
        """Halts execution until device has finished its tasks."""
        torch.cuda.synchronize()


    def memset(self, allocation, value, size):
        """This method must implement setting the memory to a value on the device.
        Not implemented: Python DSLs usually do not perform explicit memory handling."""
        pass


    def memcpy_dtoh(self, dest, src):
        """This method must implement a device to host copy.
        Not implemented: Python DSLs usually do not perform explicit memory handling."""
        pass


    def memcpy_htod(self, dest, src):
        """This method must implement a host to device copy.
        Not implemented: Python DSLs usually do not perform explicit memory handling."""
        pass


    def copy_constant_memory_args(self, cmem_args):
        raise NotImplementedError("Generic Python does not support constant memory")


    def copy_shared_memory_args(self, smem_args):
        raise NotImplementedError("Generic Python does not support shared memory")


    def copy_texture_memory_args(self, texmem_args):
        raise NotImplementedError("Generic Python does not support texture memory")


    def refresh_memory(self, gpu_memory, host_arguments, should_sync):
        """Refresh the GPU memory with the untouched host arguments. We overwrite the standard function
        because Python DSLs do usually do not manage memory explicitely"""

        for i, host_arg in enumerate(host_arguments):
            if should_sync[i]:
                gpu_arg = gpu_memory[i]

                # Scalar Python type
                if isinstance(gpu_arg, (int, float, bool)):
                    gpu_memory[i] = host_arg
                elif type(gpu_arg) in vars(builtins).values():
                    gpu_memory[i] = host_arg

                # GPU tensor
                elif isinstance(gpu_arg, torch.Tensor):
                    if isinstance(host_arg, np.ndarray):
                        gpu_arg.copy_(torch.as_tensor(host_arg))
                    elif isinstance(host_arg, torch.Tensor):
                        if host_arg.is_cuda and host_arg.device != gpu_arg.device:
                            gpu_arg.copy_(host_arg.to(gpu_arg.device))  # different gpu's, no direct copy allowed
                        else:
                            gpu_arg.copy_(host_arg)
                    else:
                        # host_arg is scalar, fill into tensor
                        gpu_arg.fill_(host_arg)


    def classify_compile_exception(self, e):
        """Best effort to differentiate between a user error and a resource error.

        Only the messages of the exception and of the exceptions it was raised from are
        inspected. The traceback is not used, because the file paths and source lines in it
        contain words like "cuda" or "ast" for almost any DSL, regardless of the error.

        :param e: the caught exception
        :type e: exception

        :returns: "resource_error" , "user_error" or "unknown"
        :rtype: string
        """

        RESOURCE_PATTERNS = (
            # Shared memory
            r"shared memory",
            r"\bsmem\b",
            r"too much shared",

            # Registers / occupancy
            r"too many registers",
            r"register spill",
            r"out of registers",
            r"register usage",

            # Launch configuration
            r"invalid launch configuration",
            r"invalid configuration",
            r"invalidconfiguration",  # cudaErrorInvalidConfiguration
            r"launch_invalid_config",
            r"threads per block",
            r"block size",
            r"grid size",
            r"\bnum_warps\b",
            r"\bnum_ctas\b",
            r"no valid warp partition",
            r"no valid schedule",

            # Generic resource exhaustion
            r"too many resources",
            r"out of resources?\b",
            r"out_of_resources",  # CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
            r"exceeds maximum",
            r"exceeds limit",

            # Errors reported by the compiler toolchain
            r"\bptxas\b",
            r"\bnvcc error",
            r"\bllvm error",
            r"\bmlir error",
            r"lowering failed",
        )

        USER_ERROR_PATTERNS = (
            # Undefined / missing symbols
            r"not defined",
            r"undefined variable",
            r"without definition",
            r"unknown variable",
            r"\bunbound\b",

            # Type / shape errors
            r"type mismatch",
            r"invalid type",
            r"cannot convert",
            r"expected .* but got",
            r"incompatible types",

            # Indexing / bounds
            r"index out of bounds",
            r"out of bounds access",
            r"invalid index",

            # IR / AST construction
            r"failed to build",
            r"invalid expression",
            r"malformed",
            r"illegal operation",

            # Frontend components
            r"\btranspiler\b",
            r"\bfrontend\b",
            r"\bast\b",
        )

        # Errors in the kernel code that no tuning parameter value can cause (IndentationError is a SyntaxError)
        CODE_ERROR_TYPES = (NameError, UnboundLocalError, SyntaxError)

        # Usually user errors, but some DSLs also raise these for configurations they do not support
        USER_ERROR_TYPES = (AttributeError, TypeError)

        # Follow the chain of exceptions like Python does when printing a traceback,
        # wrappers often hide the original error in __cause__ or __context__
        chain = []
        while e is not None and all(e is not seen for seen in chain):
            chain.append(e)
            e = e.__cause__ if e.__cause__ is not None else (None if e.__suppress_context__ else e.__context__)
        msg = "\n".join(str(exc) for exc in chain).lower()

        def match_any(patterns, text):
            return any(re.search(p, text) for p in patterns)

        if any(isinstance(exc, CODE_ERROR_TYPES) for exc in chain):
            return "user_error"

        if match_any(RESOURCE_PATTERNS, msg):
            return "resource_error"

        if any(isinstance(exc, USER_ERROR_TYPES) for exc in chain):
            return "user_error"

        if match_any(USER_ERROR_PATTERNS, msg):
            return "user_error"

        return "unknown"
