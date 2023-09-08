.. toctree::
   :maxdepth: 2


Backends
========

Kernel Tuner implements multiple backends for CUDA, one for OpenCL, one for HIP, and a generic 
Compiler backend.

Selecting a backend is in most cases automatic and is done based on the kernel's programming 
language, but sometimes you'll want to specifically choose a backend.


CUDA Backends
-------------

PyCUDA is default CUDA backend in Kernel Tuner. It is comparable in feature completeness with CuPy.
Because the HIP kernel language is identical to the CUDA kernel language, HIP is included here as well.
To use HIP on nvidia GPUs, see https://github.com/jatinx/hip-on-nv.

While the PyCUDA backend expects all inputs and outputs to be Numpy arrays, the CuPy backend also 
supports cupy arrays as input and output arguments for the kernels. This gives the user more control 
over how memory is handled by Kernel Tuner. Also checks during output verification can happen 
entirely on the GPU when using only cupy arrays.

Texture memory is only supported by the PyCUDA backend, while the CuPy backend is the only one that 
support C++ signatures for the kernels. With the other backends, it is required that the kernel has 
extern "C" linkage. If not, the entire code is wrapped in an extern "C" block, which may cause issues 
if the code also contains C++ code that cannot have extern "C" linkage, including code that may be 
present in header files.

As detailed further :ref:`templates`, templated kernels are fully supported by the CuPy backend and 
limited support is implemented by Kernel Tuner to support templated kernels for the PyCUDA and 
CUDA-Python backends.


.. csv-table:: Backend feature support
  :header: Feature, PyCUDA, CuPy, CUDA-Python, HIP
  :widths: auto

  Compile kernels,        ✓,  ✓,  ✓,  ✓
  Benchmark kernels,      ✓,  ✓,  ✓,  ✓
  Observers,              ✓,  ✓,  ✓,  ✓
  Constant memory,        ✓,  ✓,  ✓,  ✓
  Dynamic shared memory,  ✓,  ✓,  ✓,  ✓
  Texture memory,         ✓,  ✗,  ✗,  ✗
  C++ kernel signature,   ✗,  ✓,  ✗,  ✗
  Templated kernels,      ✓,  ✓,  ✓,  ✗


Another important difference between the different backends is the compiler that is used. The table 
below lists which Python package is required, how the backend can be selected and which compiler is 
used to compile the kernels.


.. csv-table:: Backend usage and compiler
  :header: Feature, PyCUDA, CuPy, CUDA-Python, HIP
  :widths: auto

  Python package,      "pycuda", "cupy", "cuda-python", "pyhip-interface"
  Selected with lang=, "CUDA", "CUPY", "NVCUDA", "HIP"
  Compiler used,       "nvcc", "nvrtc", "nvrtc", "hiprtc"


