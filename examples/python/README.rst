Kernel Tuner Examples for Python DSLs
=====================================

These examples show how to use Kernel Tuner to tune kernels written in Python-based embedded
domain-specific languages (DSLs), such as Triton, Numba, CuPy (``cupyx.jit``), Warp, Taichi, CuTe,
Tilus, TileLang, and cuTile. Kernel Tuner tunes these kernels using its Python backend, see the
`documentation on backends <https://kerneltuner.github.io/kernel_tuner/stable/backends.html>`__
for more information.

To run an example, install PyTorch with CUDA support and the DSL used by the example, for example
``pip install triton``, ``pip install numba-cuda``, ``pip install cupy-cuda13x``, ``pip install warp-lang``,
``pip install taichi``, ``pip install nvidia-cutlass-dsl``, ``pip install tilus``, ``pip install tilelang``,
or ``pip install cuda-tile``. Kernel Tuner only imports a DSL when it is used.

In all examples, the kernel is passed to ``tune_kernel`` as the name of the kernel function (or
class) and the path to the Python file that contains it. Kernel Tuner detects that the kernel is
written in Python and which DSL it uses, and launches the kernel with a default call function for
that DSL. There is no need to pass the ``lang`` or ``call_function`` options, unless you want to
launch the kernel in a different way, as shown in the Warp vector add and CuTe matrix multiplication
examples.

.. note::

    Please do not use the examples as performance benchmarks.
    The examples here are created specifically to highlight certain features in Kernel Tuner.
    Please contact the developers if you are interested in benchmarking Kernel Tuner.

Below we list the example applications and the features they illustrate.

Vector Add
----------

Numba
~~~~~
[`numba_vec_add.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/numba_vec_add.py>`__]
 - tune the thread block size of a Numba CUDA kernel just like a CUDA kernel
 - pass NumPy arrays as kernel arguments, which Kernel Tuner copies to the GPU

Triton
~~~~~~
[`triton_vec_add.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/triton_vec_add.py>`__]
 - tune a tunable parameter that is also a kernel argument (a ``tl.constexpr``)
 - pass PyTorch tensors that already live on the GPU as kernel arguments
 - use ``run_kernel`` to run a single configuration of a Python kernel

Tilus
~~~~~
[`tilus_vec_add.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/tilus_vec_add.py>`__]
 - tune a class-based kernel, where the tunable parameters are attributes of the class
 - tune the number of warps per thread block

TileLang
~~~~~~~~
[`tilelang_vec_add.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/tilelang_vec_add.py>`__]
 - tune the arguments of a kernel factory function decorated with ``@tilelang.jit``
 - pass a fixed value to the kernel factory using a tunable parameter with a single value

CuTe
~~~~
[`cute_vec_add.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/cute_vec_add.py>`__]
 - tune a value that is assigned inside the ``@cute.jit`` function that launches the kernel

cuTile
~~~~~~
[`cutile_vec_add.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/cutile_vec_add.py>`__]
 - tune the tile size of a tile-based kernel
 - use the tile size as block size, so Kernel Tuner computes a grid with one block per tile

Warp
~~~~
[`warp_vec_add.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/warp_vec_add.py>`__]
 - tune the amount of work per thread, a value that is assigned inside the kernel
 - use a helper function (``@wp.func``) that is called by the kernel
 - pass a user-defined call function to control how the kernel is launched

Matrix Multiplication
---------------------

CuPy
~~~~
[`cupy_matmul.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/matmul/cupy_matmul.py>`__]
 - tune a ``cupyx.jit.rawkernel`` with 2-dimensional thread blocks
 - use the restrictions option to limit the search to valid thread block sizes

Numba
~~~~~
[`numba_matmul.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/matmul/numba_matmul.py>`__]
 - tune a basic kernel and a kernel that uses shared memory tiling and register blocking
 - tell Kernel Tuner to compute the grid dimensions for 2D thread blocks with tiling
 - use restrictions that relate multiple tunable parameters to each other
 - use Bayesian Optimization to search a large search space

Taichi
~~~~~~
[`taichi_matmul.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/matmul/taichi_matmul.py>`__]
 - tune the block size that Taichi uses to parallelize a loop (``ti.loop_config``)

Warp
~~~~
[`warp_matmul.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/matmul/warp_matmul.py>`__]
 - use the special tunable parameters ``block_dim`` and ``dim`` to set the thread block size and
   launch dimensions of a Warp kernel

Triton
~~~~~~
[`triton_matmul.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/matmul/triton_matmul.py>`__]
 - tune a basic kernel and the optimized kernel from the Triton tutorials
 - tune the Triton compiler options ``num_warps`` and ``num_stages``
 - use Bayesian Optimization to search a large search space

Tilus
~~~~~
[`tilus_matmul.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/matmul/tilus_matmul.py>`__]
 - tune the tile sizes and number of warps of class-based kernels
 - use Bayesian Optimization to search a large search space

TileLang
~~~~~~~~
[`tilelang_matmul.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/matmul/tilelang_matmul.py>`__]
 - tune the tile sizes, number of threads, and number of software pipelining stages
 - use Bayesian Optimization to search a large search space

CuTe
~~~~
[`cute_matmul.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/matmul/cute_matmul.py>`__]
 - tune a naive kernel and the Ampere tensor core GEMM from the CuTe DSL examples
 - tune a class-based kernel, where the tunable parameters are attributes of the class
 - use restrictions to exclude configurations that exceed the shared memory or that the kernel does not support
 - pass a user-defined call function that creates the tensor layouts required by the kernel

cuTile
~~~~~~
[`cutile_matmul.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/python/matmul/cutile_matmul.py>`__]
 - tune the tile sizes of a tile-based kernel that uses tensor cores
 - tune the size of groups of tiles to improve L2 cache reuse, using a 1-dimensional grid
 - tell Kernel Tuner to compute a grid with one block per tile of the output matrix
