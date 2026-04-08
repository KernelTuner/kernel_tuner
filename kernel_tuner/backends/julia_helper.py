"""Helper functions for Julia backend detection and interaction.

We might want to consider moving this to a utility module or Julia package as it can be useful.
"""

import subprocess
from json import JSONDecodeError
from json import loads as json_loads
from re import search as regex_search
from warnings import warn

# Map name → Julia module and device-selection calls
backend_map = {
    "CUDA": {
        "pkg": "CUDA",
        "module": "CUDA",
        "device_select": lambda d: f"CUDA.device!({d})",
        "name": "CUDA.name(CUDA.device())",
        "max_threads": "CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)",
        "capability": "CUDA.capability(CUDA.device())",
        "GPUArrayType": "CuArray",
    },
    "AMD": {
        "pkg": "AMDGPU",
        "module": "ROCBackend",
        "device_select": lambda d: f"AMDGPU.device!(AMDGPU.devices()[{d}])",
        "name": "HIP.name(HIP.device())",
        "max_threads": "HIP.attribute(dev, HIP.hipDeviceAttributeMaxThreadsPerBlock)",
        "capability": None,
        "GPUArrayType": "ROCArray",
    },
    "INTEL": {
        "pkg": "oneAPI",
        "module": "oneAPI",
        "device_select": lambda d: f"devices(first(drivers()))[{d}]",
        "name": "oneAPI.name(oneAPI.device())",
        "max_threads": "oneAPI.compute_properties(oneAPI.device()).maxTotalGroupSize",
        "capability": None,
        "GPUArrayType": "oneArray",
    },
    "METAL": {
        "pkg": "Metal",
        "module": "Metal",
        "device_select": lambda d: "Metal.device!(Metal.device())",  # only single device support in Metal.jl
        "name": "Metal.name(Metal.device())",
        "max_threads": "Int(Metal.device().maxThreadsPerThreadgroup.width)",
        "capability": None,
        "GPUArrayType": "MtlArray",
    },
}


def detect_julia_gpu_backends():
    """Detect the Julia backends available."""
    available_backends = []
    for backend_name in ["CUDA", "AMD", "METAL", "INTEL"]:
        if backend_name == "CUDA":
            try:
                subprocess.check_output("nvidia-smi")
                available_backends.append(backend_name)
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        elif backend_name == "AMD":
            try:
                subprocess.check_output("rocm-smi")
                available_backends.append(backend_name)
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        elif backend_name == "METAL":
            try:
                output = subprocess.check_output("system_profiler -json SPDisplaysDataType".split())
                json_output = json_loads(output)["SPDisplaysDataType"]
                for gpu in json_output:
                    if "spdisplays_mtlgpufamilysupport" in gpu:
                        supported = gpu["spdisplays_mtlgpufamilysupport"].lower()
                        if "metal" in supported:
                            version = regex_search(r".*metal([\d.]+)", supported).group(1)
                            if float(version) < 3:
                                warn(
                                    f"Metal backend detected, but {supported} < 3. "
                                    "Metal.jl requires Metal version 3 or higher."
                                )
                            else:
                                available_backends.append(backend_name)
            except (FileNotFoundError, subprocess.CalledProcessError, JSONDecodeError):
                pass
        elif backend_name == "INTEL":
            # this can give false positives for other backends too, so skip if we've already detected another backend
            if len(available_backends) > 0:
                continue
            try:
                subprocess.check_output(
                    "ls /dev/dri/by-path/".split()
                )  # not a perfect check but should work in most cases
                available_backends.append(backend_name)
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
    return available_backends
