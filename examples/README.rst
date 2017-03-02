Kernel Tuner Examples
=====================

Most of the examples show how to use the Kernel Tuner to tune a specific
CUDA, OpenCL, or C kernel.

Except for `test\_vector\_add.py <cuda/test_vector_add.py>`__ which
shows how to use run\_kernel to implement a test that you can run with
nosetests or pytest to test your CUDA or OpenCL kernels from Python.

Below we list the example applications and the features they illustrate.

Vector Add [`CUDA <cuda/vector_add.py>`__] [`OpenCL <opencl/vector_add.py>`__] [`C <c/vector_add.py>`__]
--------------------------------------------------------------------------------------------------------

-  use the Kernel Tuner to tune a simple kernel

Stencil [`CUDA <cuda/stencil.py>`__] [`OpenCL <opencl/stencil.py>`__]
---------------------------------------------------------------------

-  use a 2-dimensional problem domain with 2-dimensional thread blocks
   in a simple and clean example

Matrix Multiplication [`CUDA <cuda/matmul.py>`__] [`OpenCL <opencl/matmul.py>`__]
---------------------------------------------------------------------------------

-  pass a filename instead of a string with code
-  use 2-dimensional thread blocks and tiling in both dimensions
-  tell the Kernel Tuner to compute the grid dimensions for 2D thread
   blocks with tiling
-  use the restrictions option to limit the search to only valid
   configurations

Convolution
-----------

There are several different examples centered around the convolution
kernel [`CUDA <cuda/convolution.cu>`__]
[`OpenCL <opencl/convolution.cl>`__]

| **convolution.py [`CUDA <cuda/convolution.py>`__]
  [`OpenCL <opencl/convolution.py>`__]** - use tunable parameters for
  tuning for multiple input sizes
| - pass constant memory arguments to the kernel
| - write output to a json file

| **sepconv.py [`CUDA <cuda/sepconv.py>`__]
  [`OpenCL <opencl/sepconv.py>`__]** - use the convolution kernel for
  separable filters
| - write output to a csv file using Pandas

| **convolution\_correct.py [`CUDA <cuda/convolution_correct.py>`__]
  [`OpenCL <opencl/convolution_correct.py>`__]** - use run\_kernel to
  compute a reference answer
| - verify the output of every benchmarked kernel

**convolution\_streams.py [`CUDA <cuda/convolution_streams.py>`__]** -
allocate page-locked host memory from Python - overlap transfers to and
from the GPU with computation - tune parameters in the host code in
combination with those in the kernel - use the lang="C" option and set
compiler options - pass a list of filenames instead of strings with
kernel code

Reduction [`CUDA <cuda/reduction.py>`__] [`OpenCL <opencl/reduction.py>`__]
---------------------------------------------------------------------------

-  use vector types and shuffle instructions (shuffle is only available
   in CUDA)
-  tune the number of thread blocks the kernel is executed with

Sparse Matrix Vector Multiplication [`CUDA <cuda/spmv.py>`__]
-------------------------------------------------------------

-  use scipy to compute a reference answer and verify all benchmarked
   kernels
-  express that the number of thread blocks depends on the values of
   tunable parameters
