import torch

def call_tilus(kernel_function, args, kwargs):
    kernel_function(*args, **kwargs) 

def call_triton(kernel_function, args, kwargs, grid, threads, params):
    if "num_warps" in params.keys():
        kwargs["num_warps"] = params["num_warps"]
    if "num_stages" in params.keys():
        kwargs["num_stages"] = params["num_stages"]
    
    torch.cuda.nvtx.range_push("kt call")
    kernel_function[grid](*args, **kwargs)
    torch.cuda.nvtx.range_pop()

def call_tilelang(kernel_function, args, kwargs):
    compiled_kernel = kernel_function(**kwargs)
    compiled_kernel(*args)

def call_numba(kernel_function, args, kwargs, grid, threads):
    from numba import cuda
    
    numba_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            numba_args.append(cuda.as_cuda_array(arg))
        else:
            numba_args.append(arg)
    kernel_function[grid, threads](*args, **kwargs)