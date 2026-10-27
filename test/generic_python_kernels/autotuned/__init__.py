"""Vector add kernels in Python DSLs decorated with kernel_tuner.autotune, used by test_decorator.py.

Each DSL lives in its own module, because Kernel Tuner copies all imports of the kernel source file into the
module it generates for each configuration. Every module provides:

- ``AUTOTUNE``: the arguments of the autotune decorator of the kernel
- ``vector_add``: the decorated kernel
- ``launch(kernel, c, a, b, n)``: launches an autotuned kernel with the launch syntax of the DSL, the arrays are
  PyTorch CUDA tensors that are converted to the array type of the DSL where needed
"""
