.. toctree::
   :maxdepth: 2


Design documentation
====================

This section provides detailed information about the design and internals 
of the Kernel Tuner. **This information is mostly relevant for developers.**

The Kernel Tuner is designed to be extensible and support 
different search and execution strategies. The current architecture of 
the Kernel Tuner can be seen as:

.. image:: architecture.png
   :width: 500pt

At the top we have the kernel code and the Python script that tunes it, 
which uses any of the main functions exposed in the user interface.

The strategies are responsible for iterating over and searching through 
the search space. The default strategy is ``brute_force``, which 
iterates over all valid kernel configurations in the search space. 
``random_sample`` simply takes a random sample of the search space. More 
advanced strategies are continuously being implemented and improved in 
Kernel Tuner. The full list of supported strategies and how to use these
is explained in the :doc:`user-api`, see the options ``strategy`` and
``strategy_options``.

The runners are responsible for compiling and benchmarking the kernel 
configurations selected by the strategy. The sequential runner is currently
the only supported runner, which does exactly what its name says. It compiles 
and benchmarks configurations using a single sequential Python process.
Other runners are foreseen in future releases.

The runners are implemented on top of the core, which implements a
high-level *Device Interface*,
which wraps all the functionality for compiling and benchmarking
kernel configurations based on the low-level *Device Function Interface*.
Currently, we have 
five different implementations of the device function interface, which 
basically abstracts the different backends into a set of simple 
functions such as ``ready_argument_list`` which allocates GPU memory and 
moves data to the GPU, and functions like ``compile``, ``benchmark``, or 
``run_kernel``. The functions in the core are basically the main 
building blocks for implementing runners.

The observers are explained in :ref:`observers`.

At the bottom, the backends are shown. 
PyCUDA, CuPy, cuda-python, PyOpenCL and PyHIP are for tuning either CUDA, OpenCL, or HIP kernels.
The C 
Functions implementation can actually call any compiler, typically NVCC 
or GCC is used. There is limited support for tuning Fortran kernels. 
This backend was created not just to be able to tune C 
functions, but in particular to tune C functions that in turn launch GPU kernels.

The rest of this section contains the API documentation of the modules 
discussed above. For the documentation of the user API see the 
:doc:`user-api`.


Strategies
----------

Strategies are explained in :ref:`optimizations`.

Many of the strategies use helper functions that are collected in ``kernel_tuner.strategies.common``.

kernel_tuner.strategies.common
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.common
    :members:


Runners
-------

kernel_tuner.runners.sequential.SequentialRunner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.runners.sequential.SequentialRunner
    :special-members: __init__
    :members:

kernel_tuner.runners.sequential.SimulationRunner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.runners.simulation.SimulationRunner
    :special-members: __init__
    :members:


Device Interfaces
-----------------

kernel_tuner.core.DeviceInterface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.core.DeviceInterface
    :special-members: __init__
    :members:

kernel_tuner.backends.pycuda.PyCudaFunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.backends.pycuda.PyCudaFunctions
    :special-members: __init__
    :members:

kernel_tuner.backends.cupy.CupyFunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.backends.cupy.CupyFunctions
    :special-members: __init__
    :members:

kernel_tuner.backends.nvcuda.CudaFunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.backends.nvcuda.CudaFunctions
    :special-members: __init__
    :members:

kernel_tuner.backends.opencl.OpenCLFunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.backends.opencl.OpenCLFunctions
    :special-members: __init__
    :members:

kernel_tuner.backends.c.CFunctions
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.backends.c.CFunctions
    :special-members: __init__
    :members:

kernel_tuner.backends.hip.HipFunctions
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.backends.hip.HipFunctions
    :special-members: __init__
    :members:


Util Functions
--------------

kernel_tuner.util
~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.util
    :members:

