
A simple CUDA/OpenCL kernel tuner in Python
====================================
[![Build Status](https://api.travis-ci.org/benvanwerkhoven/kernel_tuner.svg?branch=master)](https://travis-ci.org/benvanwerkhoven/kernel_tuner)
[![Codacy Badge](https://api.codacy.com/project/badge/grade/016dc85044ab4d57b777449d93275608)](https://www.codacy.com/app/b-vanwerkhoven/kernel_tuner)
[![Codacy Badge](https://api.codacy.com/project/badge/coverage/016dc85044ab4d57b777449d93275608)](https://www.codacy.com/app/b-vanwerkhoven/kernel_tuner)

The goal of this project is to provide a - as simple as possible - tool 
for tuning CUDA and OpenCL kernels. This implies that any CUDA or OpenCL 
kernel can be tuned without requiring extensive changes to the original 
kernel code.

A very common problem in GPU programming is that some combination of 
thread block dimensions and other kernel parameters, like tiling or 
unrolling factors, results in dramatically better performance than other 
kernel configurations. The goal of auto-tuning is to automate the 
process of finding the best performing configuration for a given device.

This kernel tuner aims that you can directly use the tuned kernel
without introducing any new dependencies. The tuned kernels can
afterwards be used independently of the programming environment, whether
that is using C/C++/Java/Fortran or Python doesn't matter.

The kernel_tuner module currently only contains main one function which
is called tune_kernel to which you pass at least the kernel name, a string
containing the kernel code, the problem size, a list of kernel function
arguments, and a dictionary of tunable parameters. There are also a lot
of optional parameters, for a complete list see the full documentation of
[tune_kernel](http://benvanwerkhoven.github.io/kernel_tuner/sphinxdoc/html/details.html).

Documentation
-------------
The full documentation is available [here](http://benvanwerkhoven.github.io/kernel_tuner/sphinxdoc/html/index.html).

Installation
------------
To install from PyPi, install with:  
 * `pip install kernel_tuner` 
  
To install from the source:   
 * clone the repository  
    `git clone git@github.com:benvanwerkhoven/kernel_tuner.git`  
 * change into the top-level directory  
    `cd kernel_tuner`  
 * install using  
    `pip install .`

Dependencies
------------
Python 2.7 or Python 3.5

PyCuda and/or PyOpenCL (https://mathema.tician.de/software/)
 * To tune CUDA kernels  
    `pip install pycuda`
 * To tune OpenCL kernels  
    `pip install pyopencl`

Example usage
-------------
The following shows a simple example for tuning a CUDA kernel:

```python
kernel_string = """
__global__ void vector_add(float *c, float *a, float *b, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
"""

size = 10000000
problem_size = (size, 1)

a = numpy.random.randn(size).astype(numpy.float32)
b = numpy.random.randn(size).astype(numpy.float32)
c = numpy.zeros_like(b)
n = numpy.int32(size)
args = [c, a, b, n]

tune_params = dict()
tune_params["block_size_x"] = [128+64*i for i in range(15)]

tune_kernel("vector_add", kernel_string, problem_size, args, tune_params)
```
The exact same Python code can be used to tune an OpenCL kernel:
```python
kernel_string = """
__kernel void vector_add(__global float *c, __global float *a, __global float *b, int n) {
    int i = get_global_id(0);
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
"""
```
Or even just a C function, see the example [here](https://github.com/benvanwerkhoven/kernel_tuner/blob/master/examples/c/vector_add.py).

You can find these and many - more extensive - example codes, in the [examples](https://github.com/benvanwerkhoven/kernel_tuner/blob/master/examples/) directory.
See the [full documentation](http://benvanwerkhoven.github.io/kernel_tuner/sphinxdoc/html/index.html) for several highly detailed tutorial-style explanations of example kernels and the scripts to tune them.

Tuning host and kernel code
---------------------------
It is also possible to tune for combinations of tunable parameters in both host and kernel code. This allows for a number of powerfull things, such as tuning the number of streams for a kernel that uses CUDA Streams or OpenCL Command Queues to overlap transfers between host and device with kernel execution. This can be done in combination with tuning the parameters inside the kernel code. See the [convolution_streams example code](https://github.com/benvanwerkhoven/kernel_tuner/blob/master/examples/) and the [documentation](http://benvanwerkhoven.github.io/kernel_tuner/sphinxdoc/html/hostcode.html) for a detailed explanation of the kernel tuner Python script.

Correctness verification
------------------------
Optionally, you can let the kernel tuner verify the output of every kernel it compiles and benchmarks,
by passing an `answer` list. This list matches the list of arguments to the kernel, but contains the expected output of the kernel. Input arguments are replaced with None.
```python
answer = [a+b, None, None]  # the order matches the arguments (in args) to the kernel
tune_kernel("vector_add", kernel_string, problem_size, args, tune_params, answer=answer)
```

Contribution guide
------------------
The kernel tuner follows the Google Python style guide, with Sphinxdoc docstrings for module public functions. If you want to
contribute to the project please fork it, create a branch including your addition, and create a pull request.

The tests use relative imports and can be run directly after making
changes to the code. To run all tests use `nosetests` in the main directory.
To run the examples after code changes, you need to run `pip install --upgrade .` in the main directory.
Documentation is generated by typing `make html` in the doc directory, the contents
of `doc/build/html/` should then be copied to `sphinxdoc` directory of the `gh-pages` branch.

Before creating a pull request please ensure the following:
* You have written unit tests to test your additions
* All unit tests pass
* The examples still work and produce the same (or better) results
* The code is compatible with both Python 2.7 and Python 3.5
* An entry about the change or addition is created in CHANGELOG.md

Contributing authors so far:
* Ben van Werkhoven
* Berend Weel

Related work
------------
You may also like [CLTune](https://github.com/CNugteren/CLTune) by Cedric
Nugteren. CLTune is a C++ library for kernel tuning and supports various
advanced features like machine learning to optimize the time spent on tuning
kernels.
