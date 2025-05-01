import sys
import subprocess
import shutil
import os

import pytest

try:
    import pycuda.driver as drv

    drv.init()
    pycuda_present = True
except Exception:
    pycuda_present = False

try:
    import pynvml

    pynvml_present = True
except ImportError:
    pynvml_present = False

try:
    import pyopencl

    opencl_present = True
    if "namespace" in str(sys.modules["pyopencl"]):
        opencl_present = False
    if len(pyopencl.get_platforms()) == 0:
        opencl_present = False
except Exception:
    opencl_present = False

gcc_present = shutil.which("g++") is not None
gfortran_present = shutil.which("gfortran") is not None
openmp_present = "libgomp" in subprocess.getoutput(["ldconfig -p | grep libgomp"])
openacc_present = shutil.which("nvc++") is not None

try:
    import cupy

    cupy.cuda.Device(
        0
    ).attributes  # triggers exception if there are no CUDA-capable devices
    cupy_present = True
except Exception:
    cupy_present = False

try:
    import cuda

    cuda_present = True
except Exception:
    cuda_present = False

try:
    from hip import hip
    hip.hipDriverGetVersion()
    hip_present = True
except (ImportError, RuntimeError):
    hip_present = False

skip_if_no_pycuda = pytest.mark.skipif(
    not pycuda_present, reason="PyCuda not installed or no CUDA device detected"
)
skip_if_no_pynvml = pytest.mark.skipif(not pynvml_present, reason="NVML not installed")
skip_if_no_cupy = pytest.mark.skipif(
    not cupy_present, reason="CuPy not installed or no CUDA device detected"
)
skip_if_no_cuda = pytest.mark.skipif(
    not cuda_present, reason="NVIDIA CUDA not installed"
)
skip_if_no_opencl = pytest.mark.skipif(
    not opencl_present, reason="PyOpenCL not installed or no OpenCL device detected"
)
skip_if_no_gcc = pytest.mark.skipif(not gcc_present, reason="No gcc on PATH")
skip_if_no_gfortran = pytest.mark.skipif(
    not gfortran_present, reason="No gfortran on PATH"
)
skip_if_no_openmp = pytest.mark.skipif(not openmp_present, reason="No OpenMP found")
skip_if_no_openacc = pytest.mark.skipif(not openacc_present, reason="No nvc++ on PATH")
skip_if_no_hip = pytest.mark.skipif(not hip_present, reason="No HIP Python found or no HIP device detected")


def skip_backend(backend: str):
    if backend.upper() == "CUDA" and not pycuda_present:
        pytest.skip("PyCuda not installed or no CUDA device detected")
    elif backend.upper() == "CUPY" and not cupy_present:
        pytest.skip("CuPy not installed or no CUDA device detected")
    elif backend.upper() == "NVCUDA" and not cuda_present:
        pytest.skip("NVIDIA CUDA not installed")
    elif backend.upper() == "OPENCL" and not opencl_present:
        pytest.skip("PyOpenCL not installed or no OpenCL device detected")
    elif backend.upper() == "C" and not gcc_present:
        pytest.skip("No g++ on PATH")
    elif backend.upper() == "FORTRAN" and not gfortran_present:
        pytest.skip("No gfortran on PATH")
    elif backend.upper() == "OPENACC" and not openacc_present:
        pytest.skip("No nvc++ on PATH")
    elif backend.upper() == "HIP" and not hip_present:
        pytest.skip("HIP Python not installed")
