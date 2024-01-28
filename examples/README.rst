Kernel Tuner Examples
=====================

Most of the examples show how to use Kernel Tuner to tune a
CUDA, OpenCL, or C kernel, while demonstrating a particular usecase of Kernel Tuner.

Except for `test\_vector\_add.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/test_vector_add.py>`__  and 
`test\_vector\_add_parameterized.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/test_vector_add_parameterized.py>`__,
which show how to write tests for GPU kernels with Kernel Tuner.

.. note::

    Please do not use the examples as performance benchmarks.
    The examples here are created specifically to highlight certain features in Kernel Tuner.
    Please contact the developers if you are interested in benchmarking Kernel Tuner.

Below we list the example applications and the features they illustrate.

Vector Add
----------
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/vector_add.py>`__] [`CUDA-C++ <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda-c++/vector_add.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/vector_add.py>`__] [`C <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/c/vector_add.py>`__] [`Fortran <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/fortran/vector_add.py>`__]
 - use Kernel Tuner to tune a simple kernel

Stencil
-------
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/stencil.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/stencil.py>`__]
 -  use a 2-dimensional problem domain with 2-dimensional thread blocks in a simple and clean example

Matrix Multiplication
---------------------
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/matmul.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/matmul.py>`__]
 -  pass a filename instead of a string with code
 -  use 2-dimensional thread blocks and tiling in both dimensions
 -  tell Kernel Tuner to compute the grid dimensions for 2D thread blocks with tiling
 -  use the restrictions option to limit the search to only valid configurations
 -  use a user-defined performance metric like GFLOP/s

Convolution
-----------
There are several different examples centered around the convolution
kernel [`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/convolution.cu>`__]
[`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/convolution.cl>`__]

convolution.py
~~~~~~~~~~~~~~
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/convolution.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/convolution.py>`__]
 - use tunable parameters for tuning for multiple input sizes
 - pass constant memory arguments to the kernel
 - write output to a json file

sepconv.py
~~~~~~~~~~
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/sepconv.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/sepconv.py>`__]
 - use the convolution kernel for separable filters
 - write output to a csv file using Pandas

convolution\_correct.py
~~~~~~~~~~~~~~~~~~~~~~~
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/convolution_correct.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/convolution_correct.py>`__]
 - use run\_kernel to compute a reference answer
 - verify the output of every benchmarked kernel

convolution\_streams.py
~~~~~~~~~~~~~~~~~~~~~~~
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/convolution_streams.py>`__]
 - allocate page-locked host memory from Python
 - overlap transfers to and from the GPU with computation
 - tune parameters in the host code in combination with those in the kernel
 - use the lang="C" option and set compiler options
 - pass a list of filenames instead of strings with kernel code

Reduction
---------
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/reduction.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/reduction.py>`__]
 - use vector types and shuffle instructions (shuffle is only available in CUDA)
 - tune the number of thread blocks the kernel is executed with
 - tune the partial loop unrolling factor of a for-loop
 - tune pipeline that consists of two kernels
 - tune with custom output verification function

Sparse Matrix Vector Multiplication
-----------------------------------
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/spmv.py>`__]
 -  use scipy to compute a reference answer and verify all benchmarked kernels
 -  express that the number of thread blocks depends on the values of tunable parameters

Point-in-Polygon
----------------
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/pnpoly.py>`__]
 -  overlap transfers with device mapped host memory
 -  tune on different implementations of an algorithm

ExpDist
-------
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/expdist.py>`__]
 -  in-thread block 2D reduction using CUB library
 -  C++ in CUDA kernel code
 -  tune multiple kernels in pipeline

Code Generator
--------------
[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/vector_add_codegen.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/vector_add_codegen.py>`__]
 - use a Python function as a code generator
