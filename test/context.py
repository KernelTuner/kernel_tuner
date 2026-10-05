import shutil
import subprocess
import sys
import ctypes.util
import os
from os import environ
import importlib.util

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
openmp_present = ctypes.util.find_library('gomp') is not None
openacc_present = shutil.which("nvc++") is not None
running_on_ci = any([environ.get(CI, "false").lower() == "true" for CI in ["GITHUB_ACTIONS", "TRAVIS", "CIRCLECI", "GITLAB_CI"]])

try:
    import cupy

    cupy.cuda.Device(0).attributes  # triggers exception if there are no CUDA-capable devices
    cupy_present = True
except Exception:
    cupy_present = False

try:
    import cuda

    print(cuda)
    cuda_present = True
except Exception:
    cuda_present = False

try:
    from hip import hip

    hip.hipDriverGetVersion()
    hip_present = True
except (ImportError, RuntimeError):
    hip_present = False

try:
    import botorch
    import torch

    bayes_opt_botorch_present = True
except ImportError:
    bayes_opt_botorch_present = False

try:
    import gpytorch
    import torch

    bayes_opt_gpytorch_present = True
except ImportError:
    bayes_opt_gpytorch_present = False

try:
    import torch
    gen_python_torch_present = torch.cuda.is_available()
except ImportError:
    gen_python_torch_present = False


def python_dsl_present(module, subpackage=None):
    """Check if a Python DSL is installed without importing it, some DSLs are slow to import."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    if subpackage is None:
        return True
    return any(os.path.isdir(os.path.join(loc, subpackage)) for loc in spec.submodule_search_locations or [])


numba_cuda_present = python_dsl_present("numba_cuda")
warp_present = python_dsl_present("warp")
taichi_present = python_dsl_present("taichi")
cute_present = python_dsl_present("cutlass", "cute")
triton_present = python_dsl_present("triton")
tilus_present = python_dsl_present("tilus")
tilelang_present = python_dsl_present("tilelang")
cutile_present = python_dsl_present("cuda", "tile")

try:
    import pyatf

    pyatf_present = True
except ImportError:
    pyatf_present = False

try:
    import skopt

    skopt_present = True
except ImportError:
    skopt_present = False

try:
    import pymoo
    pymoo_present = True
except ImportError:
    pymoo_present = False

try:
    julia_present = importlib.util.find_spec("juliacall") is not None
except ImportError:
    julia_present = False

try:
    from autotuning_methodology.report_experiments import get_strategy_scores

    methodology_present = True
except ImportError:
    methodology_present = False

skip_if_no_pycuda = pytest.mark.skipif(not pycuda_present, reason="PyCuda not installed or no CUDA device detected")
skip_if_no_pynvml = pytest.mark.skipif(not pynvml_present, reason="NVML not installed")
skip_if_no_cupy = pytest.mark.skipif(not cupy_present, reason="CuPy not installed or no CUDA device detected")
skip_if_no_cuda = pytest.mark.skipif(not cuda_present, reason="NVIDIA CUDA not installed")
skip_if_no_opencl = pytest.mark.skipif(not opencl_present, reason="PyOpenCL not installed or no OpenCL device detected")
skip_if_no_gcc = pytest.mark.skipif(not gcc_present, reason="No gcc on PATH")
skip_if_no_gfortran = pytest.mark.skipif(not gfortran_present, reason="No gfortran on PATH")
skip_if_no_julia = pytest.mark.skipif(not shutil.which("julia") or not julia_present, reason="No Julia on PATH or juliacall not installed")
skip_if_no_openmp = pytest.mark.skipif(not openmp_present, reason="No OpenMP found")
skip_if_no_openacc = pytest.mark.skipif(not openacc_present, reason="No nvc++ on PATH")
skip_if_no_bayesopt_gpytorch = pytest.mark.skipif(
    not bayes_opt_gpytorch_present, reason="Torch and GPyTorch not installed"
)
skip_if_no_bayesopt_botorch = pytest.mark.skipif(
    not bayes_opt_botorch_present, reason="Torch and BOTorch not installed"
)
skip_if_no_hip = pytest.mark.skipif(not hip_present, reason="No HIP Python found")
skip_if_no_pyatf = pytest.mark.skipif(not pyatf_present, reason="PyATF not installed")
skip_if_no_skopt = pytest.mark.skipif(not skopt_present, reason="scikit-optimize not installed")
skip_if_no_methodology = pytest.mark.skipif(not methodology_present, reason="Autotuning Methodology not found")
skip_if_no_pymoo = pytest.mark.skipif(not pymoo_present, reason="No PyMOO found")
skip_if_no_torch = pytest.mark.skipif(not gen_python_torch_present, reason="Torch not installed or no CUDA device")
skip_if_no_numba_cuda = pytest.mark.skipif(not numba_cuda_present, reason="numba-cuda not installed")
skip_if_no_warp = pytest.mark.skipif(not warp_present, reason="Warp not installed")
skip_if_no_taichi = pytest.mark.skipif(not taichi_present, reason="Taichi not installed")
skip_if_no_cute = pytest.mark.skipif(not cute_present, reason="CuTe DSL not installed")
skip_if_no_triton = pytest.mark.skipif(not triton_present, reason="Triton not installed")
skip_if_no_tilus = pytest.mark.skipif(not tilus_present, reason="Tilus not installed")
skip_if_no_tilelang = pytest.mark.skipif(not tilelang_present, reason="TileLang not installed")
skip_if_no_cutile = pytest.mark.skipif(not cutile_present, reason="cuTile not installed")


def skip_backend(backend: str):
    if backend.upper() in ("CUDA", "PYCUDA") and not pycuda_present:
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
    elif backend.upper() == "JULIA" and not shutil.which("julia"):
        pytest.skip("No Julia on PATH")
    elif backend.upper() == "GENERIC_PYTHON" and not gen_python_torch_present:
        pytest.skip("Torch not installed or no CUDA device")
