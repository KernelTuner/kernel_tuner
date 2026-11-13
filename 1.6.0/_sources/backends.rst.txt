.. toctree::
   :maxdepth: 2

.. _backends:

Backends
========

Kernel Tuner implements multiple backends for CUDA, one for OpenCL, one for HIP, and a generic 
Compiler backend.

Selecting a backend is in most cases automatic and is done based on the kernel's programming 
language, but sometimes you'll want to specifically choose a backend.


CUDA Backends
-------------

Kernel Tuner automatically selects a CUDA backend when it detects CUDA as the language of the
kernel to be tuned. Depending on whether the required dependencies are installed it first checks
if it could use the ``cuda-python`` backend, then ``cupy``, then PyCUDA. This behavior can be overwritten
by manually selecting a backend with the ``lang=`` option. 
Because the HIP kernel language is identical to the CUDA kernel language, HIP is included here as well.
To use HIP on nvidia GPUs, see https://github.com/jatinx/hip-on-nv. HIP must be indicated manually
using the ``lang=`` option.

Kernel inputs and outputs can be handeld slightly differently by different backends, but the default
way to pass arguments to the kernel is through Numpy arrays, which is supported by all backends.
For example, while the PyCUDA backend expects all inputs and outputs to be Numpy arrays, the CuPy backend also 
supports cupy arrays as input and output arguments for the kernels. This gives the user more control 
over how memory is handled by Kernel Tuner. Also checks during output verification can happen 
entirely on the GPU when using only cupy arrays.

Texture memory is only supported by the PyCUDA backend. However, some other limitations apply to the PyCUDA
backend, for example support for kernels with C++ signatures. PyCUDA requires that the kernel has 
extern "C" linkage. If not, the entire code is wrapped in an extern "C" block, which may cause issues 
if the code also contains C++ code that cannot have extern "C" linkage, including code that may be 
present in header files.

Templated kernels and kernels with C++ signatures are supported by the cuda-python, CuPy, and HIP backends. 
As detailed further in :ref:`templates`, Kernel Tuner has limited support for templated kernels
when using the PyCUDA backend.


.. csv-table:: Backend feature support
  :header: Feature, PyCUDA, CuPy, CUDA-Python, HIP
  :widths: auto

  Compile kernels,        ✓,  ✓,  ✓,  ✓
  Benchmark kernels,      ✓,  ✓,  ✓,  ✓
  Observers,              ✓,  ✓,  ✓,  ✓
  Constant memory,        ✓,  ✓,  ✓,  ✓
  Dynamic shared memory,  ✓,  ✓,  ✓,  ✓
  Texture memory,         ✓,  ✗,  ✗,  ✗
  C++ kernel signature,   ✗,  ✓,  ✓,  ✓
  Templated kernels,      ✓,  ✓,  ✓,  ✓


Another important difference between the different backends is the compiler that is used. The table 
below lists which Python package is required, how the backend can be selected and which compiler is 
used to compile the kernels. Note that backends relying on ``nvrtc`` can be more strict about
including headers and also having host code in the code that is passed to the compiler. When using
``cuda-python`` it is generally recommended to use an environment variable called ``CUDA_HOME`` that
points to the root directory of the CUDA installation. For example, by adding 
``export CUDA_HOME=/usr/local/cuda`` to your ``.bashrc`` file.


.. csv-table:: Backend usage and compiler
  :header: Feature, PyCUDA, CuPy, CUDA-Python, HIP
  :widths: auto

  Python package,      "pycuda", "cupy", "cuda-python", "hip-python"
  Selected with lang=, "CUDA", "CUPY", "NVCUDA", "HIP"
  Compiler used,       "nvcc", "nvrtc", "nvrtc", "hiprtc"


