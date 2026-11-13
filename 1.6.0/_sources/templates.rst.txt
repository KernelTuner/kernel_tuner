.. highlight:: python
    :linenothreshold: 5

.. _templates:

Templated kernels
-----------------

It is quite common in CUDA programming to write kernels that use C++ templates. This can be very useful when writing code that can 
work for several types, for example floats and doubles. However, the use of C++ templates makes it slightly more difficult to 
directly integrate the CUDA kernel into applications that are not written in C++, for example Matlab, Fortran, or Python. And 
since Kernel Tuner is written in Python, we needed to take a few extra steps to provide support for templated CUDA kernels.

Templated kernels and kernels with C++ signatures are now fully supported when using the cuda-python, cupy, and HIP backends. 
Kernel Tuner implements some limited support for templated kernels when using the PyCUDA backend. If you use templated kernels, it is
recommended that you use either the cuda-python or cupy backend.

The rest of this section explains how to use Kernel Tuner with templated kernels.
Let's first look at an example of what it's like to tune a templated kernel with Kernel Tuner.

Example
~~~~~~~

Say we have a templated CUDA kernel in a file called vector_add.cu:

.. code-block:: cuda

    template<typename T>
    __global__ void vector_add(T *c, T *a, T *b, int n) {
        auto i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }

Then the Python script to tune this kernel would be as follows:

.. code-block:: python

    import numpy
    from kernel_tuner import tune_kernel

    size = 1000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    tune_kernel("vector_add<float>", "vector_add.cu", size, args, tune_params)

What you can see is that in the Python code we specify the template instantiation to use.
Kernel Tuner will detect the use of templated kernels when the ``kernel_name`` positional
argument to ``tune_kernel`` contains a template argument.

This feature also allows users to auto-tune template parameters of the templated kernel.
We could for example define a tunable parameter:

.. code-block:: python

    tune_params["my_type"] = ["float", "double"]

and call ``tune_kernel`` using a tunable parameter inside the template arguments:

.. code-block:: python

    tune_kernel("vector_add<my_type>", "vector_add.cu", size, args, tune_params)


Support for templated kernels is supported in most backends. However, support for templates is 
limited when using the PyCUDA backend. PyCuda requires CUDA kernels to have extern C linkage, 
which means that C++ templated kernels are not supported by PyCUDA directly. To support 
templated kernels regardless of this limitation Kernel Tuner attempts to wrap the templated 
CUDA kernel by inserting a compile-time template instantiation statement and a wrapper kernel 
that calls the templated CUDA kernel, which is actually demoted to a __device__ function in 
the process. These automatic code rewrites have a real risk of breaking the code. It is 
therefore recommended to use the cuda-python or cupy backends instead. However, if you must 
use PyCUDA, to minimize the chance of errors due to Kernel Tuner's automatic code rewrites, 
it's best to isolate the templated kernel in a single source file and include it where needed 
in the larger application.
