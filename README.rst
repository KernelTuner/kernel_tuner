Kernel Tuner: A simple CUDA/OpenCL Auto-Tuner in Python
=========================================================

|Build Status| |Codacy Badge| |Codacy Badge2|

Kernel Tuner simplifies the software development of optimized and auto-tuned GPU programs, by enabling Python-based unit testing of GPU code and making it easy to develop scripts for auto-tuning GPU kernels. This also means no extensive changes and no new dependencies are required in the kernel code. The kernels can still be compiled and used as normal from any host programming language.

Kernel Tuner provides a comprehensive solution for auto-tuning GPU programs, supporting auto-tuning of user-defined parameters in both host and device code, supporting output verification of all benchmarked kernels during tuning, as well as many optimization strategies to speed up the tuning process.

Documentation
-------------

The full documentation is available
`here <http://benvanwerkhoven.github.io/kernel_tuner/index.html>`__.

Installation
------------

The easiest way to install the Kernel Tuner is using pip:

.. code-block:: bash

    pip install kernel_tuner

But you can also install from the git repository. This way you also get the
examples and the tutorials:

.. code-block:: bash

    git clone https://github.com/benvanwerkhoven/kernel_tuner.git
    cd kernel_tuner
    pip install .
    
To tune CUDA kernels:

  - First, make sure you have the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ installed
  - You can install PyCuda using ``pip install pycuda``

To tune OpenCL kernels:

  - First, make sure you have an OpenCL compiler for your intended OpenCL platform
  - You can install PyOpenCL using ``pip install pyopencl``

If you need more information about how to install the Kernel Tuner and all 
dependencies see the `installation guide 
<http://benvanwerkhoven.github.io/kernel_tuner/install.html>`__

Example usage
-------------

The following shows a simple example for tuning a CUDA kernel:

.. code:: python

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 10000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)
    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128, 256, 512]

    tune_kernel("vector_add", kernel_string, size, args, tune_params)

The exact same Python code can be used to tune an OpenCL kernel:

.. code:: python

    kernel_string = """
    __kernel void vector_add(__global float *c, __global float *a, __global float *b, int n) {
        int i = get_global_id(0);
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

The Kernel Tuner will detect the kernel language and select the right compiler and 
runtime. For every kernel in the parameter space, the Kernel Tuner will insert C 
preprocessor defines for the tunable parameters, compile, and benchmark the kernel. The 
timing results will be printed to the console, but are also returned by tune_kernel to 
allow further analysis. Note that this is just the default behavior, what and how 
tune_kernel does exactly is controlled through its many `optional arguments 
<http://benvanwerkhoven.github.io/kernel_tuner/user-api.html#kernel_tuner.tune_kernel>`__.

You can find many - more extensive - example codes, in the
`examples directory <https://github.com/benvanwerkhoven/kernel_tuner/blob/master/examples/>`__
and in the `Kernel Tuner
documentation pages <http://benvanwerkhoven.github.io/kernel_tuner/index.html>`__.

Tuning host and kernel code
---------------------------

It is possible to tune for combinations of tunable parameters in
both host and kernel code. This allows for a number of powerfull things,
such as tuning the number of streams for a kernel that uses CUDA Streams
or OpenCL Command Queues to overlap transfers between host and device
with kernel execution. This can be done in combination with tuning the
parameters inside the kernel code. See the `convolution\_streams example
code <https://github.com/benvanwerkhoven/kernel_tuner/blob/master/examples/>`__
and the
`documentation <http://benvanwerkhoven.github.io/kernel_tuner/hostcode.html>`__
for a detailed explanation of the kernel tuner Python script.

Search strategies for tuning
----------------------------

Kernel Tuner supports several strategies: brute_force (default), random_sample, minimize, basinhopping, and diff_evo (differential
evolution). Using different strategies is easy, you only need to specify to ``tune_kernel`` which strategy you would like to use, for example: ``strategy="basinhopping", method="TNC"``. For a full overview of the supported search strategies and methods please see the
`user api <http://benvanwerkhoven.github.io/kernel_tuner/user-api.html>`__.

Correctness verification
------------------------

Optionally, you can let the kernel tuner verify the output of every
kernel it compiles and benchmarks, by passing an ``answer`` list. This
list matches the list of arguments to the kernel, but contains the
expected output of the kernel. Input arguments are replaced with None.

.. code:: python

    answer = [a+b, None, None]  # the order matches the arguments (in args) to the kernel
    tune_kernel("vector_add", kernel_string, size, args, tune_params, answer=answer)

Contributing
------------

Please see the `Contributions Guide <http://benvanwerkhoven.github.io/kernel_tuner/contributing.html>`__.

Citation
--------
A scientific paper about the Kernel Tuner is in preparation, in the meantime please cite the Kernel Tuner as follows:

.. code:: latex

    @misc{
      author = {Ben van Werkhoven},
      title = {Kernel Tuner: A simple CUDA/OpenCL Auto-Tuner in Python},
      year = {2018}
    }

Related work
------------

You may also like `CLTune <https://github.com/CNugteren/CLTune>`__ by
Cedric Nugteren. CLTune is a C++ library for kernel tuning and supports
various advanced features like machine learning to optimize the time
spent on tuning kernels.

.. |Build Status| image:: https://api.travis-ci.org/benvanwerkhoven/kernel_tuner.svg?branch=master
   :target: https://travis-ci.org/benvanwerkhoven/kernel_tuner
.. |Codacy Badge| image:: https://api.codacy.com/project/badge/grade/016dc85044ab4d57b777449d93275608
   :target: https://www.codacy.com/app/b-vanwerkhoven/kernel_tuner
.. |Codacy Badge2| image:: https://api.codacy.com/project/badge/coverage/016dc85044ab4d57b777449d93275608
   :target: https://www.codacy.com/app/b-vanwerkhoven/kernel_tuner
