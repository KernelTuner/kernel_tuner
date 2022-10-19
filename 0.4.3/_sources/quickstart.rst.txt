Getting Started
===============

So you have installed Kernel Tuner! That's great! But now you'd like to get started tuning some GPU code.

Let's say we have a simple CUDA kernel stored in a file called vector_add_kernel.cu:

.. code-block:: cuda

    __global__ void vector_add(float * c, float * a, float * b, int n) {
        int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
        if ( i < n ) {
            c[i] = a[i] + b[i];
        }
    }


This kernel simply performs a point-wise addition of vectors a and b and stores the result in c.

To tune this kernel with Kernel Tuner, we are going to create the input and output data in Python using Numpy arrays.

.. code-block:: python

    import numpy as np
    import kernel_tuner

    size = 1000000
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

To tell Kernel Tuner how it should call the kernel, we can create a list in Python that should correspond to 
our CUDA kernel's argument list with the same order and types.

.. code-block::python

    args = [c, a, b, n]

So far, we have created the data structures needed by Kernel Tuner to call our kernel, but we have not yet specified what we 
want Kernel Tuner to tune in our kernel. For that, we create a dictionary that we call tune_params, in which keys correspond 
to tunable parameters in our kernel and the values are lists of values that these parameters may take.

.. code-block::python

    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128, 256, 512, 1024]

In the code above, we have inserted a key into our dictionary called "block_size_x". This is a special name for a tunable
parameter that is recognized by Kernel Tuner to denote the size of our thread block in the x-dimension. 
For a full list of special parameter names, please see the :ref:`parameter-vocabulary`.

Alright, we are all set to start calling Kernel Tuner's main function, which is called tune_kernel. 

.. code-block::python

    results, env = kernel_tuner.tune_kernel("vector_add", "vector_add_kernel.cu", size, args, tune_params)

In the above, tune_kernel takes five arguments:

 * The kernel name passed as a string
 * The filename of the kernel, also as a string
 * The ``problem_size``, which corresponds to the total number of elements/threads in our kernel
 * The argument list used to call our kernel
 * The dictionary holding our tunable parameters

What happens how, is that Kernel Tuner will copy the our kernel's input and output data to the GPU, iteratively compile and 
benchmark our kernel for every possible combination of all values of all tunable parameters (a.k.a brute force tuning), and 
return the benchmarking results as a list of dictionaries, along with an ``env`` dictionary that lists important information 
about the hardware and software in which the benchmarking took place.

This wraps up the most basic use case of Kernel Tuner. There is **a lot** more functionality, which is explained in various 
guides, examples, and feature articles.


