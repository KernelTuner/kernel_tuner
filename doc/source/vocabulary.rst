.. highlight:: python
    :linenothreshold: 50


Parameter Vocabulary
--------------------

There are certain tunable parameters that have special meaning in Kernel Tuner.
This document specifies which parameters are special and what there uses are when auto-tuning GPU kernels.

In general, it is best to avoid using these parameter names for purposes other than the ones indicated in this document.

.. code-block:: python

    kernel_tuner #is inserted by Kernel Tuner to signal the code is compiled using the tuner

    block_size_* #reserved for thread block dimensions
    grid_size_* #reserved for grid dimensions, if you want to tune these use problem_size

    compiler_opt_* #reserved for future support for tuning compiler options

    loop_unroll_factor_* #reserved for tunable parameters that specify loop unrolling factors

    nvml_* #reserved for tunable parameters and outputs related to NVML
    nvml_pwr_limit #use NVML to set power limit
    nvml_gr_clock #use NVML to set graphics clock
    nvml_mem_clock #use NVML to set memory clock


There are also a number of names that Kernel Tuner uses for reporting benchmarking results. 
Because these are reported along with the tunable parameters, it is generally a good idea to not use these names for any tunable parameters.

.. code-block:: python

    time* #reserved for time measurements

    Information that can be observed using kernel_tuner.nvml.NVMLObserver:
    nvml_energy
    nvml_power
    power_readings
    core_freq
    mem_freq
    temperature

    ps_energy  # Energy as measured by PowerSensor
    ps_power   # Power as measured by PowerSensor


