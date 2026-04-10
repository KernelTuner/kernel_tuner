"""Kernel Tuner backend for running Julia CUDA.jl kernels via JuliaCall.

This backend allows Julia kernels to be compiled, launched, and observed from Python using Kernel Tuner.

Requirements:
  pip install juliacall
  and in Julia: ] add CUDA / ROCBackend / oneAPI / Metal (will be automatically installed if not present)

Notes:
- The kernel string should contain a valid Julia GPU kernel function definition.
- The kernel name must match the Julia function to be launched.
- Currently supports CuArray and scalar arguments; constant and texture memory are not implemented.
"""

from pathlib import Path
from warnings import warn

import numpy as np

from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.julia import JuliaRuntimeObserver
from kernel_tuner.util import SkippableFailure

from .julia_helper import backend_map, detect_julia_gpu_backends

try:
    from juliacall import JuliaError
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
        self.available_backends = detect_julia_gpu_backends()
        if compiler_options is not None and len(compiler_options) == 1:
            if compiler_options[0].upper() not in self.available_backends:
                raise ValueError(
                    f"Requested Julia backend '{compiler_options[0]}' not available. "
                    f"Available backends: {self.available_backends}"
                )
            backend_name = compiler_options[0].upper()
        else:
            if len(self.available_backends) != 1:
                raise ValueError(
                    f"Multiple or no Julia backends detected: {self.available_backends}. "
                    "Please specify exactly one backend in compiler_options."
                )
            backend_name = self.available_backends[0]

        # Initialize backend attributes
        self.device = device
        self.iterations = iterations
        self.compiler_options = compiler_options or []
        self.allocations = []
        self.current_kernel = None
        self.smem_size = 0

        # Initialize Julia backend
        self.backend = None
        self.start_evt = None
        self.end_evt = None
        self.host_time = None
        self.initialize_backend(device, backend_name=backend_name)

        # setup observers
        self.observers = observers or []
        self.observers.append(
            JuliaRuntimeObserver(
                jl.Main.KernelAbstractions,
                self,
                self.backend,
                self.backend_mod,
                self.backend_mod_name,
                stream=self.stream,
                start_event=self.start_evt,
                end_event=self.end_evt,
            )
        )
        for observer in self.observers:
            observer.register_device(self)

        jl.seval(
            f"""
            global dest_tmp, src_tmp  # for memcpy_htod
            module KernelTunerHelper
                using {self.backend_mod_name}
                const kt_julia_backend = {self.backend_mod_instname}()
                const GPUArrayType = {self.GPUArrayType}
                include("{str(Path(__file__).parent / "julia_helper.jl")}")
            end
            """
        )

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
        backend_name = backend_name.upper()
        if backend_name not in backend_map:
            raise ValueError(f"Unknown backend: {backend_name}")
        info = backend_map[backend_name]

        # Ensure the package is installed
        self.check_package_and_install(info["pkg"])

        # # Set debug level if needed
        # if backend_name == "cuda":
        #     jl.seval("ENV[\"JULIA_CUDA_DEBUG\"] = \"2\"")

        # Bring module into Python
        self.backend_mod_name = info["module"]
        self.backend_mod_instname = info["module_backend"]
        jl.seval(f"using KernelAbstractions, {info['pkg']}")
        backend_mod = getattr(jl.Main, self.backend_mod_name)
        self.backend_mod = backend_mod
        jl.seval(f"tmp_arr = {info['GPUArrayType']}(Float32.(zeros(2)))")
        self.backend = jl.seval("KernelAbstractions.get_backend(tmp_arr)")
        self.GPUArrayType = info["GPUArrayType"]
        jl.seval("tmp_arr = nothing; GC.gc()")  # free temporary array

        # Select device
        try:
            if int(device) == 0 and not info["pkg"] == "CUDA":
                device = 1  # Julia uses 1-based indexing, but the CUDA backend uses 0-based so we skip that
            jl.seval(info["device_select"](int(device)))
            self.last_selected_device = device
        except Exception as e:
            raise RuntimeError(f"Failed to select Julia {info['module']} device {device}: {e}") from e

        # Query device name
        try:
            self.name = str(jl.seval(info["name"]))
        except JuliaError:
            self.name = f"{backend_name}-device-{device}"

        # Query capability if available
        if info["capability"] is not None:
            try:
                cc_tuple = jl.seval(info["capability"])
                # CUDA returns structs with major/minor fields
                self.cc = f"{cc_tuple.major}{cc_tuple.minor}"
            except JuliaError:
                self.cc = None
        else:
            self.cc = None

        # Query max threads
        try:
            self.max_threads = int(jl.seval(info["max_threads"]))
        except JuliaError:
            self.max_threads = None

        # Get the device and context
        self.backend_device = self.backend_mod.device()
        if backend_name == "CUDA":
            self.contextqueue = self.backend_mod.context
        # elif backend_name == "AMD":
        #     self.contextqueue = self.backend_mod.queue
        # elif backend_name == "INTEL":
        #     self.contextqueue = jl.seval(
        #         f"ZeCommandQueue(ZeContext(first(drivers())), devices(first(drivers()))[{int(device) + 1}]))"
        #     )
        elif backend_name == "METAL":
            self.contextqueue = self.backend_mod.MTLCommandQueue(self.backend_device)

        # Optional: common KernelAbstractions stream abstraction
        try:
            self.stream = backend_mod.get_default_stream()
        except Exception:
            self.stream = None

        # Set up stream and event attributes for observers
        if backend_name == "CUDA":
            self.stream = backend_mod.stream()
            self.start_evt = backend_mod.CuEvent
            self.end_evt = backend_mod.CuEvent
        elif backend_name == "AMD":
            self.stream = backend_mod.stream()
            self.start_evt = backend_mod.HIP.HIPEvent
            self.end_evt = backend_mod.HIP.HIPEvent
        elif backend_name == "INTEL":
            # OneAPI: no events available
            self.start_evt = None
            self.end_evt = None
        elif backend_name == "METAL":
            self.start_evt = self.start_event
            self.end_evt = self.stop_event
        else:
            raise NotImplementedError(f"Backend {backend_name} not supported in Julia backend.")

    def __del__(self):
        # drop GPUArray references to let Julia GC handle them
        try:
            for a in self.allocations:
                del a
        except Exception:
            pass
        jl.seval("GC.gc()")

    # -------------------------
    # Memory and argument setup
    # -------------------------

    def ready_argument_list(self, arguments):
        """Convert arrays to GPU Array in Julia."""
        gpu_args = []
        for arg in arguments:
            try:
                arr = self.to_gpuarray(arg)
                gpu_args.append(arr)
                self.allocations.append(arr)
            except Exception as e:
                raise RuntimeError(f"Failed to move array to GPU: {e}")
        return gpu_args

    # -------------------------
    # Compilation
    # -------------------------

    def compile(self, kernel_instance):
        """Define Julia kernel function from kernel_instance.kernel_string."""
        kernel_code = kernel_instance.kernel_string
        kernel_name = kernel_instance.name
        self.kernel_source = kernel_instance.kernel_source
        self.host_time = None  # reset host time for this kernel instance

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

        # run the kernel
        try:
            self.host_time = self.launch_kernel(func, args_tuple, params, ndrange, workgroupsize, int(self.smem_size))
        except JuliaError as e:
            raise SkippableFailure(f"Julia kernel launch failed for {params=}: {e}")

    def start_event(self):
        """Records the event that marks the start of a measurement."""
        if self.backend_mod_name == "CUDA":
            self.backend_mod.record(self.start_evt(), self.stream)
        elif self.backend_mod_name == "AMDGPU":
            self.backend_mod.HIP.record(self.start_evt(self.stream, do_record=False, timing=True))
        elif self.backend_mod_name == "Metal":
            # Because our kernel launch happens via Kernel Abstractions, we wrap our kernel between two command buffers.
            # Normally you would just use one command buffer for the actual kernel.
            jl.start_buf = self.create_metal_buffer()
            jl.seval("Metal.commit!(start_buf)")
            self.backend_mod.wait_completed(jl.start_buf)
            return float(jl.start_buf.GPUEndTime)

    def stop_event(self):
        """Records the event that marks the end of a measurement."""
        if self.backend_mod_name == "CUDA":
            self.backend_mod.record(self.end_evt(), self.stream)
        elif self.backend_mod_name == "AMDGPU":
            self.backend_mod.HIP.record(self.end_evt(self.stream, do_record=False, timing=True))
        elif self.backend_mod_name == "Metal":
            jl.end_buf = self.create_metal_buffer()
            jl.seval("Metal.commit!(end_buf)")
            self.backend_mod.wait_completed(jl.end_buf)
            return float(jl.end_buf.GPUStartTime)

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        return True  # JuliaCall synchronizes on record

    # @staticmethod
    def synchronize(self):
        try:
            jl.Main.KernelAbstractions.synchronize(self.backend)
        except JuliaError as e:
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
        except JuliaError as e:
            raise RuntimeError(f"Julia memset failed: {e}")

    @staticmethod
    def memcpy_dtoh(dest, src):
        """Perform a device to host memory copy."""
        try:
            np.copyto(dest, src)
        except JuliaError as e:
            raise RuntimeError(f"Julia memcpy_dtoh failed: {e}")

    @staticmethod
    def memcpy_htod(dest, src):
        """Perform a host to device memory copy."""
        dest_ptr = repr(jl.UInt64(jl.pointer_from_objref(dest)))
        jl.dest_tmp = dest
        jl.src_tmp = src
        jl.seval("copyto!(dest_tmp, Array(src_tmp))")
        assert dest_ptr == repr(jl.UInt64(jl.pointer_from_objref(jl.dest_tmp)))

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
                    f'{package}.jl not found in your Julia environment. Run `using Pkg; Pkg.add("{package}")` in Julia.'
                ) from e

    def create_metal_buffer(self):
        """Create a Metal buffer in the command queue."""
        try:
            buf = self.contextqueue.commandBuffer()
        except Exception:
            buf = self.backend_mod.MTLCommandBuffer(self.contextqueue)
        return buf
