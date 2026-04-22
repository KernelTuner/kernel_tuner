Metrics and Objectives
----------------------

Metrics and custom tuning objectives are two related features that are explained on this page.

.. _metrics:

Metrics
~~~~~~~

User-defined metrics serve as an easy way for the user to define their own derived results based on the measurements reported 
by Kernel Tuner, and possibly any additional observers. This allows for example to implement performance metrics, such as 
performance in floating point operations per second (e.g. GFLOP/s), or other metrics that might be more specific to the 
application, for example the number of input elements processed per second. The code below shows an example of 
user-defined metrics for a kernel that performs ``total\_flops`` GFLOP in total. User-defined 
metrics are composable, meaning that they can be defined using other user-defined metrics. That is why metrics must be
specified using an OrderedDict as they are defined in order.

.. code:: python

    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1e3)
    metrics["GFLOPS/W"] = lambda p: total_flops / p["ps_energy"]

User-defined metrics are particularly useful when the total amount of work varies between kernel configurations
and the total execution time on its own is no longer a sufficient metric to compare the performance of different kernel
configurations. This occurs in many kernels, for example in kernels that perform a reduction step, where some
part of the data reduction is left for another kernel because of synchronization. In this case, the amount of
work performed by the first kernel depends on tunable parameters such as the thread block dimensions or the
number of thread blocks.

.. _objectives:

Tuning Objectives
~~~~~~~~~~~~~~~~~

Users can specify tuning objectives other than the default optimization objective, which is kernel execution time. When using 
an optimization strategy other than exhaustive search (brute force), this objective is used to guide the optimization through 
the parameter space. The tuning objective is specified using the ``objective=`` optional parameter of ``tune_kernel()`` and
is specified as a string. This string can be a user-defined metric or a quantity reported by an observer.

In addition to specifying the name of the tuning objective, it is important for many of the optimization
strategies implemented in Kernel Tuner to know whether the objective should be minimized or maximized. Kernel
Tuner uses a list of defaults, but for some user-defined metrics the user may also need to specify the direction
of optimization. This is done by passing a boolean in the ``objective_higher_is_better=`` optional parameter of ``tune_kernel()``.


