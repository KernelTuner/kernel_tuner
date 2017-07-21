.. toctree::
   :maxdepth: 2


Design documentation
====================

This section provides detailed information about the design and internals 
of the Kernel Tuner. This information is mostly relevant for developers.

The Kernel Tuner is designed to be extensible and eventually support 
different search and execution strategies. The current architecture of 
the Kernel Tuner can be seen as:

.. image:: design.png
   :width: 500pt

At the top we have the kernel code and the Python script that tunes it, 
which uses any of the main functions exposed in the user interface.

The runners are responsible for iterating over the search space. The 
default runner is the sequential brute force runner, which does exactly 
what its name says. It iterates over the entire search space with a 
single sequential Python process. Random Sample simply takes a random 
sample of the search space.

The Noodles runner is currently being developed, which is a parallel 
runner that uses the Noodles library to parallelize a brute force 
iteration over the search space, across of a number of Python processes 
on the same node or across a number of nodes in a compute cluster.

Actually, there are many more runners that you could think of and we 
have plans for implementing them. At the moment the runners actually mix 
two concepts, one is using search strategies or machine learning to 
prune the search space and the other is parallel and distributed 
execution all the instances that need compiling and benchmarking. It is 
possible that when we have more runners we will also separate the two 
concepts within architecture of the kernel tuner.

The core layer contains all the utility functions for doing string 
manipulations and core functionality, such as compiling and benchmarking 
kernels based on the *Device Function Interface*. Currently we have 
three different implementations of the device function interface, which 
basically abstracts the different backends into a set of simple 
functions such as ``ready_argument_list`` which allocates GPU memory and 
moves data to the GPU, and functions like ``compile``, ``benchmark``, or 
``run_kernel``. The functions in the core are basically the main 
building blocks for implementing runners.

At the bottom, the three backends that we currently have are shown. 
PyCUDA and PyOpenCL are for tuning either CUDA or OpenCL kernels. The C 
Functions implementation can actually call any compiler, typically NVCC 
or GCC is used. This backend was created not just to be able to tune C 
functions, but mostly to tune C functions that also launch GPU kernels.

The rest of this section contains the API documentation of the modules 
discussed above.

kernel_tuner.strategies.brute_force
-------------------------------------------
.. automodule:: kernel_tuner.strategies.brute_force
    :members:

kernel_tuner.strategies.random_sample
-------------------------------------------
.. automodule:: kernel_tuner.strategies.random_sample
    :members:

kernel_tuner.runners.sequential.SequentialRunner
------------------------------------------------
.. autoclass:: kernel_tuner.runners.sequential.SequentialRunner
    :special-members: __init__
    :members:

kernel_tuner.core.DeviceInterface
-----------------------------------
.. autoclass:: kernel_tuner.core.DeviceInterface
    :special-members: __init__
    :members:

kernel_tuner.util
-----------------------------------
.. automodule:: kernel_tuner.util
    :members:

kernel_tuner.cuda.CudaFunctions
-------------------------------
.. autoclass:: kernel_tuner.cuda.CudaFunctions
    :special-members: __init__
    :members:

kernel_tuner.opencl.OpenCLFunctions
-----------------------------------
.. autoclass:: kernel_tuner.opencl.OpenCLFunctions
    :special-members: __init__
    :members:

kernel_tuner.c.CFunctions
-----------------------------------
.. autoclass:: kernel_tuner.c.CFunctions
    :special-members: __init__
    :members:

