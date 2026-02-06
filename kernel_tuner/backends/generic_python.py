import logging
import inspect
import copy
import traceback # for compile error handling
import re

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.generic_python import GenericPythonRuntimeObserver

try:
    import torch
except ImportError:
    torch = None


# TODO delete temp file



class GenericPythonFunctions(GPUBackend):

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        '''
        In here, everyting is generic if the language uses CUDA as backend
        '''
        if not torch: 
            logging.error("Unable to import Torch")
            raise ImportError("Unable to import Torch")

        self.device_id = torch.cuda.current_device()

        self.device_properties = torch.cuda.get_device_properties(self.device_id)
        self.name = torch.cuda.get_device_name(self.device_id)
        self.max_threads = self.device_properties.max_threads_per_multi_processor

        env = dict()
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

        # Variables to be filled in at compile time:
        self.call_function = None
        self.signature = None 
        self.gpu_kwargs = None 

        super().__init__(device=device, iterations=iterations, compiler_options=compiler_options, observers=observers)

    def ready_argument_list(self, arguments):
        '''
        The user already supplies the arguments in the correct format, because we are working with
        a Python based language anyway. TODO probably only works with torch and numpy?
        '''
        return copy.deepcopy(arguments)

    def compile(self, kernel_instance, gpu_args=None):
        logging.debug("Compiling Generic Python kernel")

        if kernel_instance.kernel_fn is None:
            raise ValueError("kernel_fn is None, currently Generic Python only supports callable kernel_source")

        if gpu_args is None:
            raise ValueError("gpu_args is None, Generic Python needs gpu args to compile the kernel")
        
        # The first time we compile, we also set the call function and the signature
        if self.call_function is None or self.signature is None:
            self.call_function = kernel_instance.kernel_source.call_function
            self.signature = kernel_instance.kernel_source.signature
        
        grid = kernel_instance.grid
        threads = kernel_instance.threads
        params = kernel_instance.params
        if inspect.isclass(kernel_instance.kernel_fn):  
            kernel_function = kernel_instance.kernel_fn()
        elif callable(kernel_instance.kernel_fn):
            # Handles functions and decroators that return callable objects
            kernel_function = kernel_instance.kernel_fn
        else:
            raise TypeError("kernel function is not a class or function")
        
        self.gpu_kwargs = self.build_gpu_kwargs(params)
       
        # Call the jit function in order to compile it
        self.synchronize()
        self.call_function(kernel_function, gpu_args, self.gpu_kwargs, grid, threads, params) 
        self.synchronize()


        return kernel_function

    def start_event(self):
        logging.debug("Start Generic Python event")
        self.start.record()

    def stop_event(self):
        logging.debug("Stop Generic Python event")
        self.end.record()

    def kernel_finished(self):
        logging.debug("Checking if kernel has finished")
        return self.end.query()

    def run_kernel(self, func, gpu_args, threads, grid, stream=None, params=None):

        # Run the kernel
        if stream is None:
            stream = self.stream

        with torch.cuda.stream(stream):
            logging.debug("Running Generic Python kernel")
            self.call_function(func, gpu_args, self.gpu_kwargs, grid, threads, params) 

    
    
    def build_gpu_kwargs(self, params=None):
        gpu_kwargs = {}
        
        if params is None:
            return gpu_kwargs
    
        for name, p in self.signature.parameters.items():
            if name in params:
                gpu_kwargs[name] = params[name]

        return gpu_kwargs
    
    
    def synchronize(self):
        torch.cuda.synchronize()

    def memset(self, allocation, value, size):
        pass

    def memcpy_dtoh(self, dest, src):
        pass

    def memcpy_htod(self, dest, src):
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
        for i, arg in enumerate(host_arguments):
            if should_sync[i]:
                gpu_memory[i] = copy.deepcopy(arg)


    def classify_compile_exception(self, e):
        """Best effort to differentiate between a user error and a resource error. Input is Exception"""

        RESOURCE_KEYWORDS = (
            # Shared memory
            "shared memory",
            "smem",
            "uses too much shared",
            "exceeds shared memory",
            "shared memory limit",

            # Registers / occupancy
            "too many registers",
            "register spill",
            "uses too many registers",
            "out of registers",

            # Launch configuration
            "invalid launch configuration",
            "invalid configuration argument",
            "threads per block",
            "block size",
            "grid size",
            "num_warps",
            "num_ctas",

            # Generic resource exhaustion
            "too many resources",
            "out of resources",
            "exceeds maximum",
            "exceeds limit",

            # Compiler-level indicators
            "ptxas error",
            "ptxas fatal",
            "nvcc error",
            "cuda error",
            "llvm error",
            "mlir error",
            "lowering failed",
        )

        USER_ERROR_KEYWORDS = (
            # Undefined / missing symbols
            "not defined",
            "undefined variable",
            "without definition",
            "unknown variable",
            "unbound",

            # Type / shape errors
            "type mismatch",
            "invalid type",
            "cannot convert",
            "expected .* but got",
            "incompatible types",

            # Indexing / bounds
            "index out of bounds",
            "out of bounds access",
            "invalid index",

            # IR / AST construction
            "failed to build",
            "invalid expression",
            "malformed",
            "illegal operation",
        )

        RESOURCE_ORIGINS = (
            "ptxas",
            "nvcc",
            "cuda",
            "llvm",
            "mlir",
            "cubin",
            "fatbin",
        )

        USER_ORIGINS = (
            "transpiler",
            "scheduler",
            "hidet",
            "frontend",
            "ast",
        )


        USER_ERROR_TYPES = (
            NameError,
            UnboundLocalError,
            AttributeError,
            TypeError,
            SyntaxError,
            IndentationError,
        )

        if isinstance(e, USER_ERROR_TYPES):
            return "user_error"
        
        
        def match_any(patterns, text):
            return any(re.search(p, text) for p in patterns)

        msg = str(e).lower()
        tb = "".join(traceback.format_tb(e.__traceback__)).lower()

        
        if match_any(RESOURCE_KEYWORDS, msg):
            return "resource_error"

        if match_any(RESOURCE_ORIGINS, msg + tb):
            return "resource_error"

        if match_any(USER_ERROR_KEYWORDS, msg):
            return "user_error"
        
        if match_any(USER_ORIGINS, msg + tb):
            return "user_error"

        return "unknown"
