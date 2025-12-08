"""Kernel Tuner backend for running Julia CUDA.jl kernels via JuliaCall.

This backend allows Julia kernels to be compiled, launched, and observed from Python using Kernel Tuner.

Requirements:
  pip install juliacall
  and in Julia: ] add CUDA

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
            raise ImportError(
                "JuliaCall not installed. Please run `pip install juliacall`."
            )

        # Ensure CUDA.jl is available
        self.check_package_and_install("CUDA")

        # Initialize CUDA events
        self.CUDA = jl.Main.CUDA
        self.stream = self.CUDA.stream()
        self.start_evt = None
        self.end_evt = None

        # Select device
        try:
            jl.seval(f"CUDA.device!({int(device)})")
            JuliaFunctions.last_selected_device = device
        except Exception as e:
            raise RuntimeError(f"Failed to set Julia CUDA device {device}: {e}")

        # Gather device info
        try:
            self.name = jl.seval("CUDA.name(CUDA.device())")
            cc_tuple = jl.seval("CUDA.capability(CUDA.device())")
            self.cc = f"{cc_tuple.major}{cc_tuple.minor}"
        except Exception as e:
            warn(f"Could not retrieve device name and compute capability from Julia CUDA: {e}")
            self.name = f"Julia-CUDA-device-{device}"
            self.cc = None
        self.max_threads = jl.seval("CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)")

        # Initialize backend attributes
        self.device = device
        self.iterations = iterations
        self.compiler_options = compiler_options or []
        self.allocations = []
        self.current_kernel = None
        self.smem_size = 0

        # setup observers
        self.observers = observers or []
        self.observers.append(JuliaRuntimeObserver(self.CUDA))
        for observer in self.observers:
            observer.register_device(self)

        # Include helper module
        jl.include(str(Path(__file__).parent / "julia_helper.jl"))

        self.to_cuarray = jl.KernelTunerHelper.to_cuarray
        self.launch_kernel = jl.KernelTunerHelper.launch_kernel

        # env info
        self.env = {
            "device_name": self.name,
            "compute_capability": self.cc,
            "iterations": iterations,
            "compiler_options": self.compiler_options,
        }

    def __del__(self):
        # drop CuArray references to let Julia GC handle them
        try:
            for a in self.allocations:
                del a
        except Exception:
            pass

    # -------------------------
    # Memory and argument setup
    # -------------------------

    def ready_argument_list(self, arguments):
        """Convert numpy arrays to CuArray in Julia."""
        gpu_args = []
        for arg in arguments:
            if isinstance(arg, np.ndarray):
                try:
                    cu = self.to_cuarray(arg)
                    gpu_args.append(cu)
                    self.allocations.append(cu)
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
                for part in stripped.split(','):
                    uses.append(part.replace("import ", "").replace("using ", "").strip())
        for package in uses:
            self.check_package_and_install(package)

        # Wrap in a module to avoid name conflicts
        module_code = f"""
module KernelTunerUserKernel
    using CUDA
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

        gx, gy, gz = (grid + (1,) * (3 - len(grid)))[:3]
        tx, ty, tz = (threads + (1,) * (3 - len(threads)))[:3]
        args_tuple = tuple(gpu_args)
        params = tuple(params.values()) # important: the order of params must match the order in the kernel definition

        try:
            self.launch_kernel(func, args_tuple, params, (int(gx), int(gy), int(gz)),
                               (int(tx), int(ty), int(tz)), int(self.smem_size))
        except Exception as e:
            raise RuntimeError(f"Julia kernel launch failed: {e}")

    def start_event(self):
        """Records the event that marks the start of a measurement."""
        self.start_evt = self.CUDA.CuEvent()
        self.CUDA.record(self.start_evt, self.stream)

    def stop_event(self):
        """Records the event that marks the end of a measurement."""
        self.end_evt = self.CUDA.CuEvent()
        self.CUDA.record(self.end_evt, self.stream)

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        return self.end_evt is not None

    @staticmethod
    def synchronize():
        try:
            jl.seval("CUDA.synchronize()")
        except Exception as e:
            raise RuntimeError(f"Julia synchronize failed: {e}")

    # -------------------------
    # Memory utilities
    # -------------------------

    @staticmethod
    def memset(allocation, value, size):
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

    @staticmethod
    def memcpy_htod(dest, src):
        try:
            jl.src_tmp = src
            jl.seval("cu_tmp = CUDA.CuArray(src_tmp)")
            cu = jl.cu_tmp
            del jl.src_tmp
            del jl.cu_tmp
            return cu
        except Exception as e:
            raise RuntimeError(f"Julia memcpy_htod failed: {e}")

    def copy_constant_memory_args(self, cmem_args):
        raise NotImplementedError("Constant memory not supported in Julia backend. Submit a feature request if needed.")

    def copy_shared_memory_args(self, smem_args):
        self.smem_size = int(smem_args.get("size", 0))

    def copy_texture_memory_args(self, texmem_args):
        raise NotImplementedError("Texture memory not supported in Julia backend. Submit a feature request if needed.")

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
                jl.seval(f"using Pkg; Pkg.add(\"{package}\")")
                jl.seval(f"import {package}")
            except Exception as e:
                raise ImportError(
                    f"{package}.jl not found in your Julia environment. "
                    f"Run `using Pkg; Pkg.add(\"{package}\)` in Julia."
                ) from e
