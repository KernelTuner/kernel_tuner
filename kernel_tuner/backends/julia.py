"""Kernel Tuner backend for running Julia CUDA.jl kernels via JuliaCall.

This backend allows Julia kernels to be compiled, launched, and observed from Python using Kernel Tuner.

Requirements:
  pip install juliacall
  and in Julia: ] add CUDA / AMDGPU / oneAPI / Metal (will be automatically installed if not present)

Notes:
- The kernel string should contain a valid Julia GPU kernel function definition.
- The kernel name must match the Julia function to be launched.
- Currently supports CuArray and scalar arguments; constant and texture memory are not implemented.
"""

import numpy as np
from warnings import warn
from pathlib import Path

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.julia import JuliaRuntimeObserver
from kernel_tuner.util import SkippableFailure

try:
    from juliacall import Main as jl
except ImportError:
    jl = None


class JuliaFunctions(GPUBackend):
    """Backend for running Julia kernels (CUDA.jl) through JuliaCall."""

    units = {"time": "ms"}
    last_selected_device = None

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """Initialize Julia backend using JuliaCall."""
        if jl is None:
            raise ImportError("JuliaCall not installed. Please run `pip install juliacall`.")

        # Initialize backend attributes
        self.device = device
        self.iterations = iterations
        self.compiler_options = compiler_options or []
        self.allocations = []
        self.current_kernel = None
        self.smem_size = 0

        # Initialize Julia backend
        self.backend = None
        self.initialize_backend(device, compiler_options["julia_backend"])
        self.start_evt = None
        self.end_evt = None

        # setup observers
        self.observers = observers or []
        self.observers.append(
            JuliaRuntimeObserver(
                jl.Main.KernelAbstractions,
                self.backend,
                self.backend_mod_name,
                stream=self.stream,
                start_event=self.start_evt,
                end_event=self.end_evt,
            )
        )
        for observer in self.observers:
            observer.register_device(self)

        # Include helper module
        jl.include(str(Path(__file__).parent / "julia_helper.jl"))

        self.to_gpuarray = jl.KernelTunerHelper.to_gpuarray
        self.launch_kernel = jl.KernelTunerHelper.launch_kernel

        # env info
        self.env = {
            "device_name": self.name,
            "compute_capability": self.cc,
            "iterations": iterations,
            "compiler_options": self.compiler_options,
        }

    def initialize_backend(self, device, backend_name):
        """Initialize for a choice of Julia backends by backend_name, one of 'cuda', 'amd', 'intel', 'metal'."""

        # Map name → Julia module and device-selection calls
        backend_map = {
            "cuda": {
                "pkg": "CUDA",
                "module": "CUDA",
                "device_select": lambda d: f"CUDA.device!({d})",
                "name": "CUDA.name(CUDA.device())",
                "max_threads": "CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)",
                "capability": "CUDA.capability(CUDA.device())",
                "GPUArrayType": "CuArray",
            },
            "amd": {
                "pkg": "AMDGPU",
                "module": "AMDGPU",
                "device_select": lambda d: f"AMDGPU.device!({d})",
                "name": "AMDGPU.name(AMDGPU.device())",
                "max_threads": "AMDGPU.device_attribute(AMDGPU.device(), :maxthreadsperblock)",
                "capability": None,
                "GPUArrayType": "ROCArray",
            },
            "intel": {
                "pkg": "oneAPI",
                "module": "oneAPI",
                "device_select": lambda d: f"oneAPI.device!({d})",
                "name": "oneAPI.name(oneAPI.device())",
                "max_threads": "oneAPI.device_attribute(oneAPI.device(), :max_work_group_size)",
                "capability": None,
                "GPUArrayType": "OneArray",
            },
            "metal": {
                "pkg": "Metal",
                "module": "Metal",
                "device_select": lambda d: "Metal.device!(Metal.device())",  # only single device support in Metal.jl
                "name": "Metal.name(Metal.device())",
                "max_threads": "Int(Metal.device().maxThreadsPerThreadgroup.width)",
                "capability": None,
                "GPUArrayType": "MtlArray",
            },
        }

        backend_name = backend_name.lower()
        if backend_name not in backend_map:
            raise ValueError(f"Unknown backend: {backend_name}")
        info = backend_map[backend_name]

        # Ensure the package is installed
        self.check_package_and_install(info["pkg"])

        # Bring module into Python
        self.backend_mod_name = info["module"]
        backend_mod = getattr(jl.Main, self.backend_mod_name)
        self.backend_mod = backend_mod
        jl.seval(f"using KernelAbstractions, {self.backend_mod_name}")
        jl.seval(f"tmp_arr = {info['GPUArrayType']}(Float32.(zeros(2)))")
        self.backend = jl.seval("KernelAbstractions.get_backend(tmp_arr)")
        self.GPUArrayType = info["GPUArrayType"]

        # Select device
        try:
            jl.seval(info["device_select"](int(device)))
            self.last_selected_device = device
        except Exception as e:
            raise RuntimeError(f"Failed to select Julia {info['module']} device {device}: {e}") from e

        # Query device name
        try:
            self.name = jl.seval(info["name"])
        except Exception:
            self.name = f"{backend_name}-device-{device}"

        # Query capability if available
        if info["capability"] is not None:
            try:
                cc_tuple = jl.seval(info["capability"])
                # CUDA returns structs with major/minor fields
                self.cc = f"{cc_tuple.major}{cc_tuple.minor}"
            except Exception:
                self.cc = None
        else:
            self.cc = None

        # Query max threads
        try:
            self.max_threads = int(jl.seval(info["max_threads"]))
        except Exception:
            self.max_threads = None

        # Get the device and context
        self.backend_device = self.backend_mod.device()
        if backend_name == "cuda":
            self.contextqueue = self.backend_mod.context
        elif backend_name in ("amd", "intel"):
            self.contextqueue = self.backend_mod.queue
        elif backend_name == "metal":
            self.contextqueue = self.backend_mod.MTLCommandQueue(self.backend_device)

        # Optional: common KernelAbstractions stream abstraction
        try:
            self.stream = backend_mod.get_default_stream()
        except Exception:
            self.stream = None

        # Set up stream and event attributes for observers
        if backend_name == "cuda":
            self.start_evt = backend_mod.CuEvent()
            self.end_evt = backend_mod.CuEvent()
            self.stream = backend_mod.stream()
        elif backend_name == "amd":
            self.start_evt = backend_mod.ROCEvent()
            self.end_evt = backend_mod.ROCEvent()
            self.stream = backend_mod.default_stream()
        elif backend_name == "intel":
            # OneAPI: no events available
            self.start_evt = None
            self.end_evt = None
        elif backend_name == "metal":
            self.start_evt = self.start_event
            self.end_evt = self.stop_event

    def __del__(self):
        # drop GPUArray references to let Julia GC handle them
        try:
            for a in self.allocations:
                del a
        except Exception:
            pass

    # -------------------------
    # Memory and argument setup
    # -------------------------

    def ready_argument_list(self, arguments):
        """Convert numpy arrays to GPU Array in Julia."""
        gpu_args = []
        for arg in arguments:
            if isinstance(arg, np.ndarray):
                try:
                    arr = self.to_gpuarray(arg)
                    gpu_args.append(arr)
                    self.allocations.append(arr)
                except Exception as e:
                    raise RuntimeError(f"Failed to move array to GPU: {e}")
            else:
                gpu_args.append(arg)
        return gpu_args

    # -------------------------
    # Compilation
    # -------------------------

    def compile(self, kernel_instance):
        """Define Julia kernel function from kernel_instance.kernel_string."""
        kernel_code = kernel_instance.kernel_string
        kernel_name = kernel_instance.name
        self.kernel_source = kernel_instance.kernel_source

        # Extract all 'using' statements and check for required packages
        uses = []
        for line in kernel_code.splitlines():
            stripped = line.strip()
            # iterate over multiple using/import statements
            if stripped.startswith("using ") or stripped.startswith("import "):
                for part in stripped.split(","):
                    uses.append(part.replace("import ", "").replace("using ", "").strip())
        for package in uses:
            self.check_package_and_install(package)

        # Wrap in a module to avoid name conflicts
        module_code = f"""
module KernelTunerUserKernel
    using {self.backend_mod_name}
    {kernel_code}
end
        """
        try:
            jl.seval(module_code)
            self.current_kernel = jl.seval(f"KernelTunerUserKernel.{kernel_name}")
            return self.current_kernel
        except Exception as e:
            raise SkippableFailure(f"Failed to compile Julia kernel: {e} \n{module_code}")

    # -------------------------
    # Kernel launch and timing
    # -------------------------

    def run_kernel(self, func, gpu_args, threads, grid, stream=None, params=None):
        """Launch a compiled Julia kernel."""
        if func is None:
            func = self.current_kernel
        if func is None:
            raise RuntimeError("No Julia kernel compiled or provided.")

        args_tuple = tuple(gpu_args)
        params = tuple(params.values())  # important: the order of params must match the order in the kernel definition

        # prepare ndrange and workgroupsize
        remove_trailing_ones = lambda tup: tup[
            : len(tup) - next((int(i) for i, x in enumerate(reversed(tup)) if x != 1), len(tup))
        ]
        ndrange = remove_trailing_ones(grid)
        ndrange = (1,) if len(ndrange) == 0 else ndrange
        workgroupsize = remove_trailing_ones(threads)
        workgroupsize = (1,) if len(workgroupsize) == 0 else workgroupsize

        try:
            self.launch_kernel(func, args_tuple, params, ndrange, workgroupsize, int(self.smem_size))
        except Exception as e:
            raise RuntimeError(f"Julia kernel launch failed: {e}")

    def start_event(self):
        """Records the event that marks the start of a measurement."""
        if self.backend_mod_name in ("CUDA", "AMDGPU"):
            self.backend_mod.record(self.start_evt, self.stream)
        elif self.backend_mod_name == "Metal":
            # Because our kernel launch happens via Kernel Abstractions, we wrap our kernel between two command buffers.
            # Normally you would just use one command buffer for the actual kernel.
            jl.start_buf = self.create_metal_buffer()
            jl.seval("Metal.commit!(start_buf)")
            self.backend_mod.wait_completed(jl.start_buf)
            return float(jl.start_buf.GPUEndTime)  # or kernelEndTime?

    def stop_event(self):
        """Records the event that marks the end of a measurement."""
        if self.backend_mod_name in ("CUDA", "AMDGPU"):
            self.backend_mod.record(self.end_evt, self.stream)
        elif self.backend_mod_name == "Metal":
            jl.end_buf = self.create_metal_buffer()
            jl.seval("Metal.commit!(end_buf)")
            self.backend_mod.wait_completed(jl.end_buf)
            return float(jl.end_buf.GPUStartTime)  # or kernelStartTime?

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        return True  # JuliaCall synchronizes on record

    # @staticmethod
    def synchronize(self):
        try:
            jl.Main.KernelAbstractions.synchronize(self.backend)
        except Exception as e:
            raise RuntimeError(f"Julia synchronize failed: {e}")

    # -------------------------
    # Memory utilities
    # -------------------------

    @staticmethod
    def memset(allocation, value, size):
        raise NotImplementedError("memset not yet implemented for Julia backend.")
        try:
            jl.allocation_tmp = allocation
            jl.seval(f"CUDA.fill!(allocation_tmp, {int(value)})")
            del jl.allocation_tmp
        except Exception as e:
            raise RuntimeError(f"Julia memset failed: {e}")

    @staticmethod
    def memcpy_dtoh(dest, src):
        try:
            jl.src_tmp = src
            jl.seval("host_tmp = Array(src_tmp)")
            host = np.array(jl.host_tmp)
            np.copyto(dest, host)
            del jl.src_tmp
            del jl.host_tmp
        except Exception as e:
            raise RuntimeError(f"Julia memcpy_dtoh failed: {e}")

    # @staticmethod
    def memcpy_htod(dest, src):
        raise NotImplementedError("memcpy_htod not yet implemented for Julia backend.", dest, src)
        try:
            jl.src_tmp = src
            jl.seval(f"arr_tmp = {self.GPUArrayType}(src_tmp)")
            arr_tmp = jl.arr_tmp
            del jl.src_tmp
            del jl.arr_tmp
            return arr_tmp
        except Exception as e:
            raise RuntimeError(f"Julia memcpy_htod failed: {e}")

    def copy_constant_memory_args(self, cmem_args):
        raise NotImplementedError(
            "Constant memory not yet supported in Julia backend. Submit a feature request if needed."
        )

    def copy_shared_memory_args(self, smem_args):
        raise NotImplementedError(
            "Shared memory not yet supported in Julia backend. Submit a feature request if needed."
        )
        self.smem_size = int(smem_args.get("size", 0))

    def copy_texture_memory_args(self, texmem_args):
        raise NotImplementedError(
            "Texture memory not yet supported in Julia backend. Submit a feature request if needed."
        )

    # -------------------------
    # Helper functions
    # -------------------------

    def check_package_and_install(self, package):
        """Checks if the Julia package is available, and installs it if not."""
        try:
            jl.seval(f"import {package}")
        except Exception:
            try:
                warn(f"{package}.jl not found, attempting to install it directly.")
                jl.seval(f'using Pkg; Pkg.add("{package}")')
                jl.seval(f"import {package}")
            except Exception as e:
                raise ImportError(
                    f"{package}.jl not found in your Julia environment. "
                    f'Run `using Pkg; Pkg.add("{package}")` in Julia.'
                ) from e

    def create_metal_buffer(self):
        """Create a Metal buffer in the command queue."""
        try:
            buf = self.contextqueue.commandBuffer()
        except Exception:
            buf = self.backend_mod.MTLCommandBuffer(self.contextqueue)
        return buf
