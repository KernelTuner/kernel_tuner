"""Helper functions for Julia backend detection and interaction.

We might want to consider moving this to a utility module or Julia package as it can be useful.
"""

import os
import shutil
import subprocess
from glob import glob
from json import JSONDecodeError
from json import loads as json_loads
from re import search as regex_search
from warnings import warn

# Map name → Julia module and device-selection calls
backend_map = {
    "CUDA": {
        # CUDACore is the core of CUDA.jl (>= 6) without the CUDA libraries (cuBLAS, cuFFT, cuSOLVER etc.), which the
        # backend does not need. NVML is a separate, small package, used to query the device UUID.
        "pkg": "CUDACore",
        "extra_pkgs": ["NVML"],
        "module": "CUDACore",
        "module_backend": "CUDABackend",
        "helper_ext": "cuda",
        "device_select": lambda d: f"CUDACore.device!({d})",
        "name": "CUDACore.name(CUDACore.device())",
        "max_threads": "CUDACore.attribute(CUDACore.device(), CUDACore.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)",
        "capability": "CUDACore.capability(CUDACore.device())",
        "uuid": lambda d: f"NVML.uuid(NVML.Device({d}))",
        "pci_bus_id": None,
        "GPUArrayType": "CuArray",
        "supported_float_types": ["Float16", "BFloat16", "Float32", "Float64"],
    },
    "AMD": {
        "pkg": "AMDGPU",
        "module": "AMDGPU",
        "helper_ext": "amdgpu",
        "module_backend": "ROCBackend",
        "device_select": lambda d: f"AMDGPU.device!(AMDGPU.devices()[{d}])",
        "name": "AMDGPU.HIP.name(AMDGPU.HIP.device())",
        "max_threads": "AMDGPU.HIP.attribute(AMDGPU.HIP.device(), AMDGPU.HIP.hipDeviceAttributeMaxThreadsPerBlock)",
        "capability": None,
        "uuid": None,
        "pci_bus_id": None,
        "GPUArrayType": "ROCArray",
        "supported_float_types": ["Float16", "BFloat16", "Float32", "Float64"],
    },
    "INTEL": {
        "pkg": "oneAPI",
        "module": "oneAPI",
        "helper_ext": "oneapi",
        "module_backend": "oneAPIBackend",
        "device_select": lambda d: f"device!(devices(first(drivers()))[{d}])",
        "name": "oneAPI.name(oneAPI.device())",
        "max_threads": "oneAPI.compute_properties(oneAPI.device()).maxTotalGroupSize",
        "capability": None,
        "uuid": None,
        "pci_bus_id": None,
        "GPUArrayType": "oneArray",
        "supported_float_types": ["Float16", "BFloat16", "Float32", "Float64"],
    },
    "METAL": {
        "pkg": "Metal",
        "module": "Metal",
        "helper_ext": "metal",
        "module_backend": "MetalBackend",
        "device_select": lambda d: "Metal.device!(Metal.device())",  # only single device support in Metal.jl
        "name": "Metal.device().name",
        "max_threads": "Int(Metal.device().maxThreadsPerThreadgroup.width)",
        "capability": None,
        "uuid": None,
        "pci_bus_id": None,
        "GPUArrayType": "MtlArray",
        "supported_float_types": ["Float16", "BFloat16", "Float32"],
    },
    "CPU": {
        "pkg": None,
        "module": "CPU",
        "helper_ext": None,
        "module_backend": "CPU",
        "device_select": lambda d: "nothing",
        "name": "CPU",
        "max_threads": "1024",  # arbitrary as CPU doesn't have a max threads per block concept
        "capability": None,
        "uuid": None,
        "pci_bus_id": None,
        "supported_float_types": ["Float16", "Float32", "Float64"],
        "GPUArrayType": "Array",
    },
}


def detect_julia_gpu_backends():
    """Detect the Julia backends available."""
    available_backends = []
    if julia_backend_available_cuda():
        available_backends.append("CUDA")
    if julia_backend_available_amd():
        available_backends.append("AMD")
    if julia_backend_available_metal():
        available_backends.append("METAL")
    if len(available_backends) == 0:
        # this can give false positives for other backends too, so skip if we've already detected another backend
        if julia_backend_available_intel():
            available_backends.append("INTEL")

    available_backends.append("CPU")  # always add CPU backend last
    return available_backends


def julia_backend_available_cuda():
    """Check if CUDA backend is available."""
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def julia_backend_available_amd():
    """Check if AMD backend is available."""
    try:
        subprocess.check_output("rocm-smi")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def julia_backend_available_metal():
    """Check if Metal backend is available."""
    try:
        output = subprocess.check_output("system_profiler -json SPDisplaysDataType".split())
        json_output = json_loads(output)["SPDisplaysDataType"]
    except (FileNotFoundError, subprocess.CalledProcessError, JSONDecodeError):
        return False
    for gpu in json_output:
        if "spdisplays_mtlgpufamilysupport" in gpu:
            supported = gpu["spdisplays_mtlgpufamilysupport"].lower()
            if "metal" in supported:
                version = regex_search(r".*metal([\d.]+)", supported).group(1)
                if float(version) < 3:
                    warn(f"Metal backend detected, but {supported} < 3. Metal.jl requires Metal version 3 or higher.")
                else:
                    return True
    return False


def julia_backend_available_intel():
    """Check if Intel backend is available. May give false positives if other backends are present."""
    try:
        subprocess.check_output("intel-smi")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            # not a perfect check but should work in most cases
            subprocess.check_output("ls /dev/dri/by-path/".split())
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False


# UUIDs of the JLLs that provide the CUDA toolkit to CUDA.jl, these hold the preferences for using a local toolkit
CUDA_TOOLKIT_JLLS = {
    "CUDA_Runtime_jll": "76a88914-d11a-5bdc-97e0-2f5a05c973a2",
    "CUDA_Compiler_jll": "d1e2174e-dfdc-576e-b43e-73b79eb1aca8",
}
# environment variable to control the use of a local CUDA toolkit when the Julia backend installs CUDACore.jl
LOCAL_CUDA_TOOLKIT_ENV = "KERNEL_TUNER_JULIA_LOCAL_CUDA_TOOLKIT"
MIN_LOCAL_CUDA_TOOLKIT_VERSION = (12, 0)


def _cuda_version_from_output(command, pattern):
    """Run a command and parse a (major, minor) CUDA version from its output, None if this fails."""
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=30)
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    match = regex_search(pattern, output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def cuda_driver_version():
    """Return the highest CUDA version supported by the NVIDIA driver as (major, minor), None if unknown."""
    return _cuda_version_from_output(["nvidia-smi"], r"CUDA Version:\s*(\d+)\.(\d+)")


def _toolkit_candidate_dirs():
    """Directories that may hold a CUDA toolkit, in the order that CUDA.jl (CUDA_Runtime_Discovery) searches them."""
    # if set, the environment variables override the discovery
    env_dirs = [os.environ[var] for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT") if os.path.isdir(os.environ.get(var, ""))]
    if env_dirs:
        return list(dict.fromkeys(env_dirs))
    dirs = []
    ptxas = shutil.which("ptxas")
    if ptxas is not None:
        dirs.append(os.path.dirname(os.path.dirname(os.path.realpath(ptxas))))
    for base in ("/usr/local/cuda", "/opt/cuda"):
        dirs.extend(sorted(glob(f"{base}-*"), reverse=True))
        dirs.append(base)
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


def _toolkit_is_complete(toolkit_dir):
    """Check that the toolkit has the components CUDA.jl needs: the compiler, runtime library and libdevice."""
    has_cudart = any(
        glob(os.path.join(toolkit_dir, pattern))
        for pattern in ("lib64/libcudart.so*", "lib/libcudart.so*", "targets/*/lib/libcudart.so*", "bin/cudart64_*.dll")
    )
    has_libdevice = bool(glob(os.path.join(toolkit_dir, "nvvm", "libdevice", "libdevice*.bc")))
    return has_cudart and has_libdevice


def find_local_cuda_toolkit():
    """Find a local CUDA toolkit that CUDA.jl can use with the installed driver.

    Returns a tuple of the toolkit directory and its (major, minor) version, or None if no compatible toolkit is found.
    """
    driver_version = cuda_driver_version()
    if driver_version is None:
        return None
    for toolkit_dir in _toolkit_candidate_dirs():
        ptxas = os.path.join(toolkit_dir, "bin", "ptxas.exe" if os.name == "nt" else "ptxas")
        if not os.path.isfile(ptxas) or not _toolkit_is_complete(toolkit_dir):
            continue
        version = _cuda_version_from_output([ptxas, "--version"], r"release (\d+)\.(\d+)")
        # like CUDA.jl, rely on minor version compatibility: the toolkit major version may not exceed the driver's
        if version is not None and MIN_LOCAL_CUDA_TOOLKIT_VERSION <= version and version[0] <= driver_version[0]:
            return toolkit_dir, version
    return None


def local_cuda_toolkit_preferences():
    """Return the juliapkg preferences that make CUDA.jl use a local CUDA toolkit instead of downloading one.

    Controlled by the environment variable ``KERNEL_TUNER_JULIA_LOCAL_CUDA_TOOLKIT``: "auto" (default) uses a compatible
    local toolkit if one is found, "yes" always uses the local toolkit, "no" lets CUDA.jl download the toolkit.
    Returns None if the toolkit should be downloaded.
    """
    setting = os.environ.get(LOCAL_CUDA_TOOLKIT_ENV, "auto").strip().lower()
    if setting in ("no", "false", "0", "off"):
        return None
    if setting not in ("yes", "true", "1", "on"):
        if setting != "auto":
            warn(f"Unknown value '{setting}' for {LOCAL_CUDA_TOOLKIT_ENV}, expected 'auto', 'yes' or 'no'. Using 'auto'.")
        toolkit = find_local_cuda_toolkit()
        if toolkit is None:
            return None
        toolkit_dir, (major, minor) = toolkit
        warn(
            f"Using the local CUDA {major}.{minor} toolkit at {toolkit_dir} for CUDA.jl instead of downloading one. "
            f"Set {LOCAL_CUDA_TOOLKIT_ENV}=no before the first use of the Julia backend to download it instead."
        )
    return {name: {"local": "true"} for name in CUDA_TOOLKIT_JLLS}
