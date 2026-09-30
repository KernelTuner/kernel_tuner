"""Vector add kernels in Python DSLs, used by test_generic_python_dsls.py.

Each DSL lives in its own module, because Kernel Tuner copies all imports of the kernel source file
into the module it generates for each configuration. Every module provides:

- ``kernel_name``: name of the kernel function or class in the module
- ``arguments(c, a, b, n)``: the kernel arguments in the order the kernel expects them
- ``tune_params(n)``: a small set of tunable parameters
- ``call_function``: the call function that launches the kernel
"""
