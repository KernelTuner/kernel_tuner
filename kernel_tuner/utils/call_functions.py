"""Default call functions that launch kernels written in Python DSLs, used with ``lang="generic_python"``.

A call function receives the kernel function with the tunable parameters of the current configuration
inserted, the kernel arguments as prepared by Kernel Tuner (PyTorch CUDA tensors and Python scalars),
and keyword arguments for the tunable parameters that are also arguments of the kernel. Optionally, it
also receives the grid and thread block dimensions computed by Kernel Tuner and the tunable parameters
of the configuration, see :func:`kernel_tuner.util.normalize_call_function`.

When ``call_function`` is not passed to ``tune_kernel`` or ``run_kernel``, the DSL of the kernel is
detected and the matching call function from this module is used.

The DSLs are optional dependencies, so they are only imported when a call function is used.
"""


def _convert_tensors(args, convert):
    """Convert the PyTorch tensors in args using convert, leave other arguments as they are."""
    import torch

    return [convert(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]


def call_triton(kernel_function, args, kwargs, grid, threads, params):
    """Launch a Triton kernel, num_warps and num_stages are taken from the tunable parameters if present."""
    for option in ("num_warps", "num_stages"):
        if option in params:
            kwargs[option] = params[option]
    kernel_function[grid](*args, **kwargs)


def call_numba(kernel_function, args, kwargs, grid, threads):
    """Launch a Numba CUDA kernel with the grid and thread block dimensions computed by Kernel Tuner."""
    from numba import cuda

    numba_args = _convert_tensors(args, cuda.as_cuda_array)
    kernel_function[grid, threads](*numba_args, **kwargs)


def call_cupyx(kernel_function, args, kwargs, grid, threads):
    """Launch a CuPy JIT (cupyx.jit.rawkernel) kernel, which only accepts positional arguments."""
    import cupy as cp

    cupy_args = _convert_tensors(args, cp.from_dlpack)
    kernel_function(grid, threads, (*cupy_args, *kwargs.values()))


def call_warp(kernel_function, args, kwargs, grid, threads, params):
    """Launch a Warp kernel.

    The launch dimensions are taken from the tunable parameter ``dim`` if present, otherwise the total
    number of threads is used. The thread block size is taken from the tunable parameter ``block_dim`` if
    present, otherwise the thread block dimensions computed by Kernel Tuner are used.
    """
    import warp as wp

    wp.init()
    warp_args = _convert_tensors(args, wp.from_torch)
    block_dim = params.get("block_dim", threads[0] * threads[1] * threads[2])
    if "dim" in params:
        dim = params["dim"]
    else:
        # launch with as many dimensions as the problem, kernels using i = wp.tid() need a 1D launch
        dim = [g * t for g, t in zip(grid, threads)]
        while len(dim) > 1 and dim[-1] == 1:
            dim.pop()
    wp.launch(kernel_function, dim=dim, inputs=[*warp_args, *kwargs.values()], block_dim=block_dim)


def call_taichi(kernel_function, args, kwargs):
    """Launch a Taichi kernel, Taichi determines the launch configuration itself."""
    kernel_function(*args, **kwargs)


def call_cute(kernel_function, args, kwargs):
    """Compile and launch a CuTe DSL kernel.

    The kernel is compiled once per configuration. Kernel Tuner compiles and benchmarks the configurations
    one by one, so only the most recently compiled kernel is kept.
    """
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    cute_args = _convert_tensors(args, from_dlpack)
    cached_function, compiled_kernel = getattr(call_cute, "_last_compiled", (None, None))
    if cached_function is not kernel_function:
        compiled_kernel = cute.compile(kernel_function, *cute_args)
        call_cute._last_compiled = (kernel_function, compiled_kernel)
    compiled_kernel(*cute_args, **kwargs)


def call_tilus(kernel_function, args, kwargs):
    """Launch a Tilus kernel, Tilus kernels are scripts that determine the launch configuration themselves."""
    kernel_function(*args, **kwargs)


def call_tilelang(kernel_function, args, kwargs):
    """Launch a TileLang kernel, the tunable parameters are passed to the kernel factory function."""
    compiled_kernel = kernel_function(**kwargs)
    compiled_kernel(*args)


def call_cutile(kernel_function, args, kwargs, grid):
    """Launch a cuTile kernel on the current PyTorch stream.

    ``ct.launch`` only accepts positional arguments, so the tunable parameters that are also kernel
    arguments are passed after the other arguments: these should be the last arguments of the kernel.
    Large tiles can take very long to compile, so the compile time is limited to 60 seconds, the
    resulting timeout error makes Kernel Tuner skip the configuration.
    """
    import cuda.tile as ct
    import torch

    with ct.compiler_timeout(60):
        ct.launch(torch.cuda.current_stream(), grid, kernel_function, (*args, *kwargs.values()))


DEFAULT_CALL_FUNCTIONS = {
    "triton": call_triton,
    "numba": call_numba,
    "cupyx": call_cupyx,
    "warp": call_warp,
    "taichi": call_taichi,
    "cute": call_cute,
    "tilus": call_tilus,
    "tilelang": call_tilelang,
    "cutile": call_cutile,
}


def get_default_call_function(dsl):
    """Return the default call function for a DSL, as detected by detect_python_dsl."""
    if dsl not in DEFAULT_CALL_FUNCTIONS:
        raise ValueError(f"No default call function for DSL {dsl}, supported are {list(DEFAULT_CALL_FUNCTIONS)}")
    return DEFAULT_CALL_FUNCTIONS[dsl]
