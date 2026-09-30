.. toctree::
   :maxdepth: 2


Backends
========

Kernel Tuner implements multiple backends for CUDA, one for OpenCL, one for HIP, a generic
Compiler backend, and a Python backend for kernels written in Python-based DSLs.

Selecting a backend is in most cases automatic and is done based on the kernel's programming 
language, but sometimes you'll want to specifically choose a backend.


CUDA Backends
-------------

For ``lang="CUDA"``, Kernel Tuner uses the CUDA-Python backend if cuda-python is installed, otherwise
CuPy, and otherwise PyCUDA. PyCUDA is comparable in feature completeness with CuPy.
Because the HIP kernel language is identical to the CUDA kernel language, HIP is included here as well.
To use HIP on nvidia GPUs, see https://github.com/jatinx/hip-on-nv.

While the PyCUDA backend expects all inputs and outputs to be Numpy arrays, the CuPy backend also 
supports cupy arrays as input and output arguments for the kernels. This gives the user more control 
over how memory is handled by Kernel Tuner. Also checks during output verification can happen 
entirely on the GPU when using only cupy arrays.

Texture memory is only supported by the PyCUDA backend, while the CuPy backend is the only one that 
support C++ signatures for the kernels. With the other backends, it is required that the kernel has 
extern "C" linkage. If not, the entire code is wrapped in an extern "C" block, which may cause issues 
if the code also contains C++ code that cannot have extern "C" linkage, including code that may be 
present in header files.

As detailed further in :ref:`templates`, templated kernels are fully supported by the CuPy backend and 
limited support is implemented by Kernel Tuner to support templated kernels for the PyCUDA and 
CUDA-Python backends.


.. csv-table:: Backend feature support
  :header: Feature, PyCUDA, CuPy, CUDA-Python, HIP
  :widths: auto

  Compile kernels,        ✓,  ✓,  ✓,  ✓
  Benchmark kernels,      ✓,  ✓,  ✓,  ✓
  Observers,              ✓,  ✓,  ✓,  ✓
  Constant memory,        ✓,  ✓,  ✓,  ✓
  Dynamic shared memory,  ✓,  ✓,  ✓,  ✓
  Texture memory,         ✓,  ✗,  ✗,  ✗
  C++ kernel signature,   ✗,  ✓,  ✗,  ✗
  Templated kernels,      ✓,  ✓,  ✓,  ✗


Another important difference between the different backends is the compiler that is used. The table 
below lists which Python package is required, how the backend can be selected and which compiler is 
used to compile the kernels.


.. csv-table:: Backend usage and compiler
  :header: Feature, PyCUDA, CuPy, CUDA-Python, HIP
  :widths: auto

  Python package,      "pycuda", "cupy", "cuda-python", "hip-python"
  Selected with lang=, "CUDA", "CUPY", "NVCUDA", "HIP"
  Compiler used,       "nvcc", "nvrtc", "nvrtc", "hiprtc"


Python Backend
--------------

The Python backend tunes kernels written in Python-based embedded domain-specific languages (DSLs).
Kernel Tuner supports the DSLs listed in the table below, examples for each DSL can be found in
`examples/python <https://github.com/kerneltuner/kernel_tuner/tree/master/examples/python>`__.

To tune a Python kernel, pass the name of the kernel function (or class) as ``kernel_name`` and the path
to the Python file that contains the kernel as ``kernel_source``. Kernel Tuner selects the Python backend
for kernels in Python files automatically, or you can select it with ``lang="generic_python"``.
The Python backend requires PyTorch with CUDA support and currently only supports NVIDIA GPUs.
The DSLs are optional dependencies, which are only imported when they are used.

.. code-block:: python

    import triton
    import triton.language as tl

    @triton.jit
    def vector_add(c_ptr, a_ptr, b_ptr, n, block_size_x: tl.constexpr):
        offsets = tl.program_id(axis=0) * block_size_x + tl.arange(0, block_size_x)
        mask = offsets < n
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(c_ptr + offsets, a + b, mask=mask)

    ...

    results, env = tune_kernel("vector_add", __file__, n, [c, a, b, n], {"block_size_x": [128, 256, 512]})


Call functions
~~~~~~~~~~~~~~

Every DSL has its own way to launch a kernel. Kernel Tuner uses a *call function* to launch the kernel
for each configuration. When ``call_function`` is not passed to ``tune_kernel`` or ``run_kernel``,
Kernel Tuner detects the DSL of the kernel from its decorators or base classes, and uses the default call
function for that DSL from :mod:`kernel_tuner.utils.call_functions`:

.. csv-table:: Supported Python DSLs
  :header: DSL, Python package, Detected by, Default launch
  :widths: auto

  Triton,             "triton",             "``@triton.jit``",               "``kernel[grid]``, ``num_warps`` and ``num_stages`` are taken from the tunable parameters"
  Numba,              "numba-cuda",         "``@numba.cuda.jit``",           "``kernel[grid, threads]``"
  CuPy,               "cupy",               "``@cupyx.jit.rawkernel``",      "``kernel(grid, threads, args)``"
  Warp,               "warp-lang",          "``@warp.kernel``",              "``wp.launch``, tunable parameters ``dim`` and ``block_dim`` set the launch dimensions"
  Taichi,             "taichi",             "``@taichi.kernel``",            "``kernel(*args)``"
  CuTe,               "nvidia-cutlass-dsl", "``@cutlass.cute.jit``",         "``cute.compile`` once per configuration"
  Tilus,              "tilus",              "subclass of ``tilus.Script``",  "``kernel(*args)``"
  TileLang,           "tilelang",           "``@tilelang.jit``",             "tunable parameters are passed to the kernel factory"
  cuTile,             "cuda-tile",          "``@cuda.tile.kernel``",         "``ct.launch`` with a compile timeout of 60 seconds"

If a kernel needs to be launched differently, you can pass your own call function. The call function
receives the following positional arguments:

- ``kernel_function``: the kernel with the values of the tunable parameters inserted
- ``args``: the list of kernel arguments, as prepared by Kernel Tuner
- ``kwargs``: a dictionary with the tunable parameters that are also arguments of the kernel

And optionally, in this order, the arguments ``grid`` and ``threads`` with the grid and thread block
dimensions computed by Kernel Tuner, and ``params``, a dictionary with the tunable parameters of the
configuration. For example:

.. code-block:: python

    def call_function(kernel_function, args, kwargs, grid):
        kernel_function[grid](*args, **kwargs)


Tunable parameters
~~~~~~~~~~~~~~~~~~

Kernel Tuner inserts the values of the tunable parameters into the kernel source code of each
configuration. Tunable parameters that are arguments of the kernel are passed to the call function in
``kwargs``. Other occurrences of tunable parameters in the kernel, and in functions defined in the same
file that are called by the kernel, are replaced by their values: reading a variable (or class attribute)
with the name of a tunable parameter returns its value, and assignments to such variables are
overridden with the value. This makes it possible to tune values that are defined inside a kernel, as well
as attributes of class-based kernels.

Only the imports, the kernel, and the functions called by the kernel are included in the source code of
each configuration. Other module-level statements and variables in the kernel source file are not
available to the kernel, unless they are tunable parameters.

Kernel Tuner computes the grid dimensions from the problem size and the block size parameters, in the
same way as for CUDA kernels. For tile-based DSLs, use ``block_size_names`` to indicate the tunable
parameters that determine the tile size, so that the grid has one block per tile. The maximum number of
threads per block is not checked by the Python backend, as tile-based DSLs determine the number of
threads themselves.


Arguments and errors
~~~~~~~~~~~~~~~~~~~~

Kernel arguments can be NumPy arrays, PyTorch tensors, or scalars. Kernel Tuner copies NumPy arrays and
PyTorch tensors to separate PyTorch tensors on the GPU, so the arguments passed by the user are not
modified, and resets outputs between kernel runs. The kernels are benchmarked using CUDA events on the
current PyTorch stream. Compiling a kernel is done by launching it once, so the compile time reported by
Kernel Tuner excludes the time of that kernel run.

Configurations that cannot be compiled or launched because they use too many resources, are too large
to compile within the time limit, or use tile sizes that the DSL does not support, are skipped. Other
errors, such as errors in the kernel code, stop the tuning process. Constant, shared, and texture memory
arguments are not supported by the Python backend.
