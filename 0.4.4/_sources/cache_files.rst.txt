.. _cache:

Cache files
===========

A very useful feature of Kernel Tuner is the ability to store benchmarking results in a cache file during tuning. You can enable cache files by 
passing any filename to the ``cache=`` optional argument of ``tune_kernel``.

The benchmark results of individual kernel configurations are appended to the cache file as Kernel Tuner is running. This also allows Kernel Tuner 
to restart a ``tune_kernel()`` session from an existing cache file, should something have terminated the previous session before the run had 
completed. This happens quite often in HPC environments when a job reservation runs out. 

Cache files enable a number of other features, such as simulations and visualizations. Simulations are useful for benchmarking optimization 
strategies. You can start a simulation by calling ``tune_kernel`` with a cache file that contains the full search space and the ``simulation=True`` option.

Cache files can be used to create visualizations of the search space. This even works while Kernel Tuner is still running. As the new results are 
coming, they are streamed to the visualization. Please see `Kernel Tuner Dashboard <https://github.com/KernelTuner/dashboard>`__.
