import sys
import pytest

import shutil

try:
    import pycuda.driver as drv
    drv.init()
    pycuda_present = True
except Exception:
    pycuda_present = False

try:
    import pyopencl
    opencl_present = True
    if 'namespace' in str(sys.modules['pyopencl']):
        opencl_present = False
    if len(pyopencl.get_platforms()) == 0:
        opencl_present = False
except Exception:
    opencl_present = False

gfortran_present = shutil.which("gfortran") is not None

try:
    import cupy
    cupy_present = True
except Exception:
    cupy_present = False

try:
    import cuda
    cuda_present = True
except Exception:
    cuda_present = False

skip_if_no_pycuda = pytest.mark.skipif(not pycuda_present, reason="PyCuda not installed or no CUDA device detected")
skip_if_no_cupy = pytest.mark.skipif(not cupy_present, reason="CuPy not installed")
skip_if_no_cuda = pytest.mark.skipif(not cuda_present, reason="NVIDIA CUDA not installed")
skip_if_no_opencl = pytest.mark.skipif(not opencl_present, reason="PyOpenCL not installed or no OpenCL device detected")
skip_if_no_gfortran = pytest.mark.skipif(not gfortran_present, reason="No gfortran on PATH")
