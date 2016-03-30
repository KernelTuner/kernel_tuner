.. kernel_tuner documentation master file, created by
   sphinx-quickstart on Tue Mar 29 15:46:32 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The kernel_tuner documentation
==============================

Contents:

.. toctree::
   :maxdepth: 2

   Introduction <self>
   details
   examples

Introduction
============
.. automodule:: kernel_tuner.kernel_tuner

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
 * PyCuda (https://mathema.tician.de/software/pycuda/)
 * A CUDA capable device


Example usage
-------------
The following shows a simple example use of the kernel tuner:

::

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

    args = [c, a, b]
    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    tune_kernel("vector_add", kernel_string, problem_size, args, tune_params)

Author
------
Ben van Werkhoven <b.vanwerkhoven@esciencenter.nl>

Copyright and License
---------------------
* Copyright 2016 Netherlands eScience Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

