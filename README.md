
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

The kernel_tuner module currently only contains one function which is called
tune_kernel to which you pass at least the kernel name, a string
containing the kernel code, the problem size, a list of kernel function
arguments, and a dictionary of tunable parameters. There are also a lot
of optional parameters, for a full list see the documentation of
tune_kernel.

Documentation
-------------
The full documentation is available [here](http://benvanwerkhoven.github.io/kernel_tuner/sphinxdoc/html/index.html).

Installation
------------
clone the repository  
    `git clone git@github.com:benvanwerkhoven/kernel_tuner.git`  
change into the top-level directory  
    `cd kernel_tuner`  
install using  
    `pip install .`

Dependencies
------------
 * PyCuda and/or PyOpenCL (https://mathema.tician.de/software/)

Example usage
-------------
The following shows a simple example use of the kernel tuner:

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
And for OpenCL:
```python
kernel_string = """
__kernel void vector_add(__global float *c, __global float *a, __global float *b, int n) {
    int i = get_global_id(0);
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
"""

size = 10000000
problem_size = (size, 1)

a = numpy.random.rand(size).astype(numpy.float32)
b = numpy.random.rand(size).astype(numpy.float32)
c = numpy.zeros_like(a)
n = numpy.int32(size)

args = [c, a, b, n]

tune_params = dict()
tune_params["block_size_x"] = [128+64*i for i in range(15)]

tune_kernel("vector_add", kernel_string, problem_size, args, tune_params)

```
More extensive examples are available in the `examples` directory

Contribution guide
------------------
The kernel tuner follows the Google Python style guide. If you want to
contribute to the project please fork it, create a branch including
your addition, and create a pull request.

Contributing authors so far:
* Ben van Werkhoven



