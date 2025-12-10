"""Module for kernel tuner cuda-python utility functions."""

import numpy as np

try:
    from cuda.bindings import driver, runtime, nvrtc
except ImportError:
    cuda = None

NVRTC_VALID_CC = np.array(["50", "52", "53", "60", "61", "62", "70", "72", "75", "80", "87", "89", "90", "90a"])


def cuda_error_check(error):
    """Checking the status of CUDA calls using the NVIDIA cuda-python backend."""
    if isinstance(error, driver.CUresult):
        if error != driver.CUresult.CUDA_SUCCESS:
            _, name = driver.cuGetErrorName(error)
            raise RuntimeError(f"CUDA Driver error: {name.decode()}")
    elif isinstance(error, runtime.cudaError_t):
        if error != runtime.cudaError_t.cudaSuccess:
            _, name = runtime.cudaGetErrorName(error)
            raise RuntimeError(f"CUDA Runtime error: {name.decode()}")
    elif isinstance(error, nvrtc.nvrtcResult):
        if error != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, desc = nvrtc.nvrtcGetErrorString(error)
            raise RuntimeError(f"NVRTC error: {desc.decode()}")


def to_valid_nvrtc_gpu_arch_cc(compute_capability: str) -> str:
    """Returns a valid Compute Capability for NVRTC `--gpu-architecture=`, as per https://docs.nvidia.com/cuda/nvrtc/index.html#group__options."""
    return max(NVRTC_VALID_CC[NVRTC_VALID_CC <= compute_capability], default="52")
