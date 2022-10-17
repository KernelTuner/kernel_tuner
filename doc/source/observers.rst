Observers
---------

To facilitate measurements of quantities other than kernel execution time, and to make it easy
for the user to control exactly what is being measured by Kernel Tuner, we have introduced the Observers
feature. In the layered software architecture of Kernel Tuner, observers act as programmable hooks to allow the
user to change or expand Kernel Tuner's benchmarking behavior at any of the lower levels. Following the observer
design pattern, observers can be used to subscribe to certain types of events and the methods implemented by the
observer will be called when the event takes place.

Kernel Tuner implements an abstract BenchmarkObserver with methods that may be overwritten by classes extending
the BenchmarkObserver class, shown below. The only mandatory method to implement
is ``get\_results``, which is used to return the resulting observations at the end of benchmarking a
particular kernel configuration and usually returns aggregated results over multiple iterations of kernel
execution. Before tuning starts, each observer is given a reference to the lower-level backend that is used for
compiling and benchmarking the kernel configurations. In this way, the observer can inspect the compiled module,
function, the state of GPU memory, or any other information in the GPU runtime.

.. autoclass:: kernel_tuner.observers.BenchmarkObserver


The PyOpenCL, PyCUDA, Cupy, and cuda-python backends support observers. Each backend also implements their own observer to
measure the runtime of kernel configurations during benchmarking. The user specifies a list of observers to use when calling Kernel
Tuner. This feature makes it easy to extend Kernel Tuner with observers for quantities other than time and the
user can easily define their own observers, without the need to modify Kernel Tuner's source code.
See for example a RegisterObserver that observes the number of
registers per thread used by the compiled kernel configuration shown below.
There are many more possible observers that could be implemented, for example an observer could be created to
track performance counters during auto-tuning..

.. code:: python

    class RegisterObserver(BenchmarkObserver):
        def get_results(self):
            return {"num_regs": self.dev.current_module.func.num_regs}


PowerSensorObserver
~~~~~~~~~~~~~~~~~~~

PowerSensor2 is a custom-built power measurement device for PCIe devices that
intercepts the device power with current sensors and transmits the data to the host over a USB connection. The
main advantage of using PowerSensor2 over the GPU's built-in power sensor is that PowerSensor2 reports
instantaneous power consumption with a very high frequency (about 2.8 KHz). PowerSensor2 comes with an
easy-to-use software library that supports various forms of power measurement. We have created a simple
interface using PyBind11\footnote{https://pybind11.readthedocs.io/en/stable/} to the PowerSensor library to make
it possible to use it from Python.

Kernel Tuner implements a PowerSensorObserver specifically for use with PowerSensor2, that can be selected by
the user to record power and/or energy consumption of kernel configurations during auto-tuning. This allows
Kernel Tuner to accurately determine the power and energy consumption of all kernel configurations it benchmarks
during auto-tuning.

.. autoclass:: kernel_tuner.observers.PowerSensorObserver


NVMLObserver
~~~~~~~~~~~~

Kernel Tuner also implements an NVMLObserver, which allows the user to observe the power usage, energy
consumption, core and memory frequencies, core voltage and temperature for all kernel configurations during
benchmarking as reported by the NVIDIA Management Library (NVML). To facilitate the interaction with
NVML, Kernel Tuner implements a thin wrapper that abstracts some of the intricacies of NVML into a more user
friendly and Pythonic interface. The NVMLObserver is implemented on top of this interface.

To ensure that the power measurements in Kernel Tuner obtained using NVML accurately reflect the power
consumption of the kernel, we have introduced a continuous benchmarking mode that takes place after the regular
iterative benchmarking process. During continuous benchmarking, the kernel is executed repeatedly for a
user-specified duration, 1 second by default. The NVMLObserver uses the continuous benchmarking mode when power or energy
measurements are requested by the user. The downside of this approach is that it significantly increases that
time it takes to benchmark different kernel configurations. However, NVML can be used for power measurements on
almost all Nvidia GPUs, so this method is much more accessible to end-users compared to solutions that require
custom hardware, such as PowerSensor2.

.. autoclass:: kernel_tuner.nvml.NVMLObserver
