import sys
import subprocess
import shutil

import pytest

try:
    import pycuda.driver as drv
    drv.init()
    cuda_present = True
except Exception:
    cuda_present = False

try:
    import pyopencl
    opencl_present = True
    if 'namespace' in str(sys.modules['pyopencl']):
        opencl_present = False
    if len(pyopencl.get_platforms()) == 0:
        opencl_present = False
except Exception:
    opencl_present = False

gcc_present = shutil.which("gcc") is not None
gfortran_present = shutil.which("gfortran") is not None
openmp_present = "libgomp" in subprocess.getoutput(["ldconfig -p | grep libgomp"])

try:
    import cupy
    cupy_present = True
except Exception:
    cupy_present = False

skip_if_no_cuda = pytest.mark.skipif(not cuda_present, reason="PyCuda not installed or no CUDA device detected")
skip_if_no_cupy = pytest.mark.skipif(not cupy_present, reason="CuPy not installed")
skip_if_no_opencl = pytest.mark.skipif(not opencl_present, reason="PyOpenCL not installed or no OpenCL device detected")
skip_if_no_gcc = pytest.mark.skipif(not gfortran_present, reason="No gcc on PATH")
skip_if_no_gfortran = pytest.mark.skipif(not gfortran_present, reason="No gfortran on PATH")
skip_if_no_openmp = pytest.mark.skipif(not gfortran_present, reason="No OpenMP found")
