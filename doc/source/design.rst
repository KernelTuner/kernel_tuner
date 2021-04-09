.. toctree::
   :maxdepth: 2


Design documentation
====================

This section provides detailed information about the design and internals 
of the Kernel Tuner. **This information is mostly relevant for developers.**

The Kernel Tuner is designed to be extensible and support 
different search and execution strategies. The current architecture of 
the Kernel Tuner can be seen as:

.. image:: design.png
   :width: 500pt

At the top we have the kernel code and the Python script that tunes it, 
which uses any of the main functions exposed in the user interface.

The strategies are responsible for iterating over and searching through 
the search space. The default strategy is ``brute_force``, which 
iterates over all valid kernel configurations in the search space. 
``random_sample`` simply takes a random sample of the search space. More 
advanced strategies currently implemented in Kernel Tuner are 
``minimize``, ``basinhopping``, and differential evolution 
(``diff_evo``). How to use these is explained in the :doc:`user-api`,
see the options ``strategy`` and ``strategy_options``.

The runners are responsible for compiling and benchmarking the kernel 
configurations selected by the strategy. The sequential runner is currently
the only supported runner, which does exactly what its name says. It compiles 
and benchmarks configurations using a single sequential Python process.
Other runners are foreseen in future releases.

The runners are implemented on top of a high-level *Device Interface*,
which wraps all the functionality for compiling and benchmarking
kernel configurations based on the low-level *Device Function Interface*.
Currently, we have 
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
discussed above. For the documentation of the user API see the 
:doc:`user-api`.



Strategies
----------

kernel_tuner.strategies.brute_force
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.brute_force
    :members:

kernel_tuner.strategies.random_sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.random_sample
    :members:

kernel_tuner.strategies.minimize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.minimize
    :members:

kernel_tuner.strategies.basinhopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.basinhopping
    :members:

kernel_tuner.strategies.diff_evo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.strategies.diff_evo
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

kernel_tuner.cuda.CudaFunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.cuda.CudaFunctions
    :special-members: __init__
    :members:

kernel_tuner.opencl.OpenCLFunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.opencl.OpenCLFunctions
    :special-members: __init__
    :members:

kernel_tuner.c.CFunctions
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: kernel_tuner.c.CFunctions
    :special-members: __init__
    :members:


Util Functions
--------------

kernel_tuner.util
~~~~~~~~~~~~~~~~~
.. automodule:: kernel_tuner.util
    :members:

