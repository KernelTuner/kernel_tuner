import torch

def call_tilus(kernel_function, args, kwargs):
    kernel_function(*args, **kwargs) 


def call_triton(kernel_function, args, kwargs, grid, threads, params):
    if "num_warps" in params.keys():
        kwargs["num_warps"] = params["num_warps"]
    if "num_stages" in params.keys():
        kwargs["num_stages"] = params["num_stages"]
    
    kernel_function[grid](*args, **kwargs)


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


def call_cupyx(kernel_function, args, kwargs, grid, threads):
    import cupy as cp

    cupy_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            cupy_args.append(cp.from_dlpack(arg))
        else:
            cupy_args.append(arg)
    kernel_function(grid, threads, tuple(cupy_args))


def call_cute(kernel_function, args, kwargs, grid, threads, params):
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    # Initialize cache if it does not exist
    if not hasattr(call_cute, "custom_cache"):
        call_cute.custom_cache = {}  

    # Convert Torch tensors to CuTe tensors with correct layout
    cute_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg_ = from_dlpack(arg)
            cute_args.append(arg_)
        else:
            cute_args.append(arg)

    # Form cache key from tuning parameters
    param_keys = sorted(params.keys())
    cache_str = type(kernel_function).__name__
    for k in param_keys:
        cache_str += "_" + str(params[k]) 
    
    # Check if kernel exists in cache. Otherwise, compile and save
    if cache_str in call_cute.custom_cache:
        compiled_kernel = call_cute.custom_cache[cache_str]
    else: 
        compiled_kernel = cute.compile(kernel_function, *cute_args)
        call_cute.custom_cache[cache_str] = compiled_kernel

    compiled_kernel(*cute_args, **kwargs)


def call_taichi(kernel_function, args, kwargs):
    kernel_function(*args, **kwargs)


def call_warp(kernel_function, args, kwargs, grid, threads, params):
    import warp as wp
    
    # Convert Torch tensors to Warp args
    warp_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            warp_args.append(wp.from_torch(arg))
        else:
            warp_args.append(arg)

    # launch kernel
    with wp.Tape() as tape:
        wp.launch_tiled(
            kernel_function,
            dim=grid,
            inputs=warp_args,
            block_dim=params["TILE_THREADS"], # We could directly take threads, but in the given example this is a constant
        )
