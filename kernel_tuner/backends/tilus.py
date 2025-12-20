import logging
import numpy as np
import inspect

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.generic_python import GenericPythonRuntimeObserver

try:
    import torch
except ImportError:
    logging.error("Torch not available")

try:
    import tilus
except ImportError:
    tilus = None
    logging.error("Unable to load Tilus")


class TilusFunctions(GPUBackend):

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        '''
        In here, everyting is generic if the language uses Torch as backend
        '''
        if not tilus or not torch:
            logging.error("Tilus or Torch not available")
            raise ImportError("Tilus or Torch not available")

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

        super().__init__(device=device, iterations=iterations, compiler_options=compiler_options, observers=observers)

    def ready_argument_list(self, arguments):
        '''
        This seems to work for Torch as backend. However, another idea for generic would be
        to let the user provide two argument lists: one with numpy and one with the actual arguments
        that will be passed to the kernel. TODO: why do we need the numpy list anyway?
        '''
        torch_args = []

        for arg in arguments:
            if isinstance(arg, torch.Tensor) and arg.dim() > 0:
                torch_args.append(arg.cuda())
            elif isinstance(arg, torch.Tensor) and arg.dim() == 0:
                scalar_value = arg.item()
                torch_args.append(scalar_value)
            elif isinstance(arg, np.ndarray):
                torch_arg = torch.from_numpy(arg)
                torch_arg_gpu = torch_arg.cuda()
                torch_args.append(torch_arg_gpu)
            elif isinstance(arg, np.generic):
                scalar_value = arg.item()
                torch_args.append(scalar_value)
            else:
                logging.warning("Unknown instance in Tilus functions")

        return torch_args

    def compile(self, kernel_instance, gpu_args=None):
        logging.debug("Compiling Tilus kernel")

        if kernel_instance.kernel_fn is None:
            raise ValueError("kernel_fn is None, currently Tilus only supports callable kernel_source")

        '''
        if gpu_args is None:
            raise ValueError("gpu_args is None, Triton needs gpu args to compile the kernel")
        TODO: find out about these GPU args
        '''

        grid = kernel_instance.grid # TODO might need to use this?
        threads = kernel_instance.threads
        params = kernel_instance.params
        kernel_function = kernel_instance.kernel_fn
        gpu_kwargs = self.build_gpu_kwargs(kernel_function, threads, params)

        # Call the jit function in order to compile it
        kernel_function(*gpu_args, **gpu_kwargs)

        return kernel_function

    def start_event(self):
        logging.debug("Start triton event")
        self.start.record()

    def stop_event(self):
        logging.debug("Stop triton event")
        self.end.record()

    def kernel_finished(self):
        logging.debug("Checking if kernel has finished")
        return self.end.query()

    def run_kernel(self, func, gpu_args, threads, grid, stream=None, params=None):
        '''
        if params is None:
            raise ValueError("params is None, Triton needs params in order to set num_warps, num_ctas, etc.")
        '''

        # Run the kernel
        if stream is None:
            stream = self.stream

        gpu_kwargs = self.build_gpu_kwargs(func, threads, params)

        with torch.cuda.stream(stream):
            logging.debug("Running Tilus kernel")
            func(*gpu_args, **gpu_kwargs)

    
    
    def build_gpu_kwargs(self, kernel_function, threads, params=None):
        gpu_kwargs = {}

        '''
        if 'BLOCK_SIZE' in jit_fn.arg_names:
            gpu_kwargs['BLOCK_SIZE'] = threads[0]

        if 'BLOCK_SIZE_X' in jit_fn.arg_names:
            gpu_kwargs['BLOCK_SIZE_X'] = threads[0]

        if 'BLOCK_SIZE_Y' in jit_fn.arg_names:
            gpu_kwargs['BLOCK_SIZE_Y'] = threads[1]

        if 'BLOCK_SIZE_Z' in jit_fn.arg_names:
            gpu_kwargs['BLOCK_SIZE_Z'] = threads[2]
        '''

        if params is None:
            return gpu_kwargs
        

        sig = inspect.signature(kernel_function)
        for name, p in sig.parameters.items():
            if name in params:
                gpu_kwargs[name] = params[name]

        '''
        # Check for Triton specific parameters
        if 'num_warps' in params:
            gpu_kwargs['num_warps'] = params['num_warps']
        if 'num_ctas' in params:
            gpu_kwargs['num_ctas'] = params['num_ctas']
        if 'num_stages' in params:
            gpu_kwargs['num_stages'] = params['num_stages']
        '''

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
        raise NotImplementedError("Tilus does not support constant memory")

    def copy_shared_memory_args(self, smem_args):
        raise NotImplementedError("Tilus does not support shared memory")

    def copy_texture_memory_args(self, texmem_args):
        raise NotImplementedError("Tilus does not support texture memory")