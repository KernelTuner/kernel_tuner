"""Kernel Tuner backend for running Julia GPU kernels (KernelAbstractions.jl) via JuliaCall.

This backend allows Julia kernels to be compiled, launched, and observed from Python using Kernel Tuner.

Requirements:
  pip install juliacall
  and in Julia: ] add CUDACore NVML / AMDGPU / oneAPI / Metal (will be automatically installed if not present)
  An installed CUDA.jl is used as is, as it contains CUDACore and NVML.
  When installing CUDACore, a compatible local CUDA toolkit is used if available instead of downloading one,
  see `KERNEL_TUNER_JULIA_LOCAL_CUDA_TOOLKIT` in julia_helper.py.

Notes:
- The kernel string should contain a valid Julia GPU kernel function definition.
- The kernel name must match the Julia function to be launched.
- Currently supports CuArray and scalar arguments; constant and texture memory are not implemented.
"""

import os
import sys
import uuid
from pathlib import Path
from warnings import warn
from dataclasses import dataclass
from time import perf_counter

import numpy as np

# Julia stops its threads for garbage collection by letting them access a protected memory page, and handles the
# resulting signal itself, but only if JuliaCall lets Julia install its signal handlers. GPU packages run Julia code on
# threads of their own (e.g. Metal.jl waits for the GPU on GPUToolbox worker threads), even if Julia was started with a
# single thread, so without this a garbage collection can crash Python with a bus error or segmentation fault.
# JuliaCall reads this setting when it is first imported, so it is set here, before any import of JuliaCall.
# This also lets Julia take over Ctrl-C, which the backend hands back to Python (see `give_interrupts_to_python`).
JULIACALL_HANDLE_SIGNALS = "PYTHON_JULIACALL_HANDLE_SIGNALS"
if "juliacall" not in sys.modules:
    os.environ.setdefault(JULIACALL_HANDLE_SIGNALS, "yes")



def give_interrupts_to_python(jl):
    """Let Ctrl-C raise a KeyboardInterrupt in Python again, after JuliaCall started Julia with signal handling.

    Julia only needs its own signal handlers for garbage collection (SIGSEGV / SIGBUS), but it also takes over the
    interrupt signal (SIGINT), so that Ctrl-C would not stop tuning. Python's handler is installed again, so that Ctrl-C
    raises a KeyboardInterrupt once the current call into Julia returns. Julia is told not to exit on Ctrl-C, in case
    its handler is still in place (Python can only install signal handlers from the main thread).
    """
    import signal
    import threading

    jl.seval("Base.exit_on_sigint(false)")
    if threading.current_thread() is threading.main_thread():
        handler = signal.getsignal(signal.SIGINT)  # Python's handler (e.g. that of a Jupyter kernel), overridden by Julia
        if handler is not None:
            signal.signal(signal.SIGINT, handler)


from kernel_tuner.backends.backend import GPUBackend
from kernel_tuner.observers.julia import JuliaRuntimeObserver
from kernel_tuner.util import SkippableFailure

from .julia_helper import CUDA_TOOLKIT_JLLS, backend_map, detect_julia_gpu_backends, local_cuda_toolkit_preferences


jl = None
JuliaError = None


@dataclass(frozen=True)
class JuliaKernel:
    """Immutable Julia kernel representation, particularly useful for caching and parallel tuning."""
    function: object
    params: tuple


class JuliaFunctions(GPUBackend):
    """Backend for running Julia kernels (CUDA.jl) through JuliaCall."""

    units = {"time": "ms"}
    last_selected_device = None

    def __init__(self, device=0, iterations=7, compiler_options=None, observers=None):
        """Initialize Julia backend using JuliaCall."""
        # lazy import to avoid mixing e.g. CUDA drivers
        # (when Julia runs Python instead, e.g. from KernelTuner.jl, JuliaCall did not start Julia and Julia handles signals)
        juliacall_config = getattr(sys.modules.get("juliacall"), "CONFIG", {})
        if juliacall_config.get("inited") and juliacall_config.get("opt_handle_signals") != "yes":
            warn(
                "JuliaCall started Julia without signal handling, "
                "which can crash Python with a bus error or segmentation fault during garbage collection. "
                f"Set the environment variable {JULIACALL_HANDLE_SIGNALS}=yes before JuliaCall is first imported."
            )
        try:
            from juliacall import JuliaError
            from juliacall import Main as jl
            global jl, JuliaError
            jl = jl
            JuliaError = JuliaError
        except ImportError:
            jl = None
        if jl is None:
            raise ImportError("JuliaCall not installed. Please run `pip install juliacall`.")
        if getattr(sys.modules.get("juliacall"), "CONFIG", {}).get("opt_handle_signals") == "yes":
            give_interrupts_to_python(jl)

        # process passed options and backends
        self.process_compiler_options(compiler_options)
        self.available_backends = detect_julia_gpu_backends()
        backend_name = self.verify_backends_with_options(compiler_options)

        # Initialize backend attributes
        self.device = device
        self.iterations = iterations
        self.compiler_options = compiler_options or []
        self.allocations = []
        self.current_kernel = None
        self.smem_size = 0

        # Event tracking for async kernel launches
        self._start_event = None
        self._end_event = None

        # Initialize Julia backend
        self.backend = None
        self.start_evt = None
        self.end_evt = None
        self.launch_kernel_time = None
        self.launch_kernel_time_start = None
        self.initialize_backend(device, backend_name=backend_name)

        # set device environment info
        self.env = {
            "device_name": self.name,
            "compute_capability": self.cc,
            "iterations": iterations,
            "compiler_options": self.compiler_options,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
        }
        # PCI domain and device ID are only known for some backends (AMD), only add them when available
        if self.pci_domain_id is not None:
            self.env["pci_domain_id"] = self.pci_domain_id
        if self.pci_device_id is not None:
            self.env["pci_device_id"] = self.pci_device_id

        # setup observers
        self.observers = observers or []
        self.observers.append(
            # TODO this single stateful default observer currently prevents parallel tuning
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

        # setup Julia module for kernel launch
        if self.backend_mod_name == "CPU":
            jl.seval(
                f"""
                global dest_tmp, src_tmp  # for memcpy_htod
                module KernelTunerHelper
                    using KernelAbstractions
                    const kt_julia_backend = CPU()
                    const GPUArrayType = {self.GPUArrayType}
                    include("{str(Path(__file__).parent / "julia_helper.jl")}")
                end
                """
            )
        else:
            jl.seval(
                f"""
                global dest_tmp, src_tmp  # for memcpy_htod
                module KernelTunerHelper
                    using {self.backend_using}
                    const kt_julia_backend = {self.backend_mod_instname}()
                    const GPUArrayType = {self.GPUArrayType}
                    include("{str(Path(__file__).parent / "julia_helper.jl")}")
                    include("{str(Path(__file__).parent / f"julia_helper_ext_{self.helper_ext}.jl")}")
                end
                """
            )

        self.launch_kernel = jl.KernelTunerHelper.launch_kernel
        self.synchronize_gpu = jl.KernelTunerHelper.synchronize_gpu
        self.to_gpuarray = jl.KernelTunerHelper.to_gpuarray

        # Metal specific functions
        self.use_metal_buffer = self.backend_mod_name == "Metal"
        if self.use_metal_buffer:
            self.metal_kernel_finished = jl.KernelTunerHelper.metal_kernel_finished
            self.metal_gpu_time_ms = jl.KernelTunerHelper.metal_gpu_time_ms

    def initialize_backend(self, device, backend_name):
        """Initialize for a choice of Julia backends by backend_name, one of 'cuda', 'amd', 'intel', 'metal'."""
        backend_name = backend_name.upper()
        if backend_name not in backend_map:
            raise ValueError(f"Unknown backend: {backend_name}")
        info = backend_map[backend_name]
        self.backend_name = backend_name
        self.helper_ext = info["helper_ext"]
        backend_pkg = info["pkg"]

        if backend_pkg is not None:
            # Ensure the package is installed
            packages = [backend_pkg] + info.get("extra_pkgs", [])
            preferences = None
            if backend_name == "CUDA":
                if self.package_installed("CUDA"):
                    # CUDA.jl contains CUDACore and NVML, use it as is (see julia_load_cuda.jl)
                    packages = []
                elif not self.package_installed(backend_pkg):
                    # CUDA.jl downloads a CUDA toolkit on installation, prefer a compatible local toolkit instead
                    preferences = local_cuda_toolkit_preferences()
            self.check_package_and_install(
                packages, allow_install=backend_name != "INTEL", preferences=preferences
            )  # avoid Intel false positives

            # # Set debug level if needed
            # if backend_name == "cuda":
            #     jl.seval("ENV[\"JULIA_CUDA_DEBUG\"] = \"2\"")

        # Bring module into Python
        self.backend_mod_name = info["module"]
        self.backend_mod_instname = info["module_backend"]
        # jl.seval(f"using KernelAbstractions, {info['module']}")
        # KernelAbstractions is a dependency of the backend packages, but must be a direct dependency to be loaded
        self.check_package_and_install("KernelAbstractions")
        jl.seval("using KernelAbstractions")
        self.backend_using = info["module"]
        if backend_pkg is not None:
            if info.get("loader") is not None:
                # the loader binds the module in Main, so other modules defined in Main use it relatively
                jl.include(str(Path(__file__).parent / info["loader"]))
                jl.seval(f"using .{info['module']}")
                self.backend_using = f"..{info['module']}"
            else:
                jl.seval(f"using {info['module']}")
        backend_mod = getattr(jl.Main, self.backend_mod_name)
        self.backend_mod = backend_mod
        jl.seval(f"tmp_arr = {info['GPUArrayType']}(Float32.(zeros(2)))")
        self.backend = jl.seval("KernelAbstractions.get_backend(tmp_arr)")
        self.GPUArrayType = info["GPUArrayType"]
        jl.seval("tmp_arr = nothing; GC.gc()")  # free temporary array

        # Select device
        try:
            if int(device) == 0 and backend_name != "CUDA":
                # Julia uses 1-based indexing, but the CUDA backend uses 0-based so we skip that
                device = 1
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

        # Query UUID and PCI bus ID if available
        self.uuid = None
        self.pci_bus_id = None
        if info["uuid"] is not None:
            try:
                self.uuid = str(jl.seval(info["uuid"](int(device))))
            except JuliaError:
                self.uuid = None
        if info["pci_bus_id"] is not None:
            try:
                self.pci_bus_id = str(jl.seval(info["pci_bus_id"](int(device))))
            except JuliaError:
                self.pci_bus_id = None
        self.pci_domain_id = None
        self.pci_device_id = None
        if backend_name == "AMD":
            self.query_amd_device_identity()

        # Get the device and context
        try:
            self.backend_device = self.backend_mod.device()
        except Exception:
            self.backend_device = None
        if backend_name == "CUDA":
            self.contextqueue = self.backend_mod.context
        # elif backend_name == "AMD":
        #     self.contextqueue = self.backend_mod.queue
        # elif backend_name == "INTEL":
        #     self.contextqueue = jl.seval(
        #         f"ZeCommandQueue(ZeContext(first(drivers())), devices(first(drivers()))[{int(device) + 1}]))"
        #     )
        elif backend_name == "METAL":
            self.contextqueue = self.backend_mod.global_queue(self.backend_device)

        self.setup_streams(backend_name)

    def query_amd_device_identity(self):
        """Query the UUID and PCI location of the selected AMDGPU device, used by e.g. the AMDSMIObserver to find the device.

        The UUID and PCI information follow the same format as the HIP backend.
        """
        try:
            uuid_bytes, pci_domain, pci_bus, pci_device = jl.seval(
                """
                let props = AMDGPU.HIP.properties(AMDGPU.HIP.device())
                    (collect(reinterpret(UInt8, collect(props.uuid.bytes))),
                     Int(props.pciDomainID), Int(props.pciBusID), Int(props.pciDeviceID))
                end
                """
            )
            self.uuid = str(uuid.UUID(bytes=bytes(int(b) for b in uuid_bytes)))
            self.pci_domain_id = int(pci_domain)
            self.pci_bus_id = int(pci_bus)
            self.pci_device_id = int(pci_device)
        except (JuliaError, ValueError) as e:
            warn(f"Could not query the UUID and PCI information of the AMDGPU device: {e}")

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
                if np.isscalar(arg):
                    # if not explicitly float64, it should be a float32 for GPU kernels
                    if isinstance(arg, float) and not isinstance(arg, np.float64):
                        arg = np.float32(arg)
                else:
                    arg = self.to_gpuarray(arg)
                gpu_args.append(arg)
                self.allocations.append(arg)
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

        # Extract all 'using' statements and check for required packages
        uses = []
        for line in kernel_code.splitlines():
            stripped = line.strip()
            # iterate over multiple using/import statements
            if stripped.startswith("using ") or stripped.startswith("import "):
                for part in stripped.split(","):
                    uses.append(part.replace("import ", "").replace("using ", "").strip())
        for package in uses:
            self.check_package_and_install(package, allow_install=False)

        # Wrap in a module to avoid name conflicts
        if self.backend_mod_name == "CPU":
            module_code = f"""
module KernelTunerUserKernel
    {kernel_code}
end
            """
        else:
            module_code = f"""
module KernelTunerUserKernel
    using {self.backend_using}
    {kernel_code}
end
            """
        try:
            jl.seval(module_code)
            function = jl.seval(f"KernelTunerUserKernel.{kernel_name}")
            return JuliaKernel(function, tuple(kernel_instance.params.values()))    # important: the order of params must match the order in the kernel definition
        except Exception as e:
            raise SkippableFailure(f"Failed to compile Julia kernel: {e} \n{module_code}")

    # -------------------------
    # Kernel launch and timing
    # -------------------------

    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
        """Launch a compiled Julia kernel."""
        self.launch_kernel_time = None  # reset host time for this kernel instance
        self.launch_kernel_time_start = None
        if func is None or not isinstance(func, JuliaKernel):
            raise RuntimeError("No Julia kernel compiled or provided.")

        args_tuple = tuple(gpu_args)
        julia_func = func.function
        params = func.params

        remove_trailing_ones = lambda tup: tup[
            : len(tup) - next((int(i) for i, x in enumerate(reversed(tup)) if x != 1), len(tup))
        ]

        # Kernel Tuner's grid is number of workgroups; KernelAbstractions' ndrange is number of global work items
        if len(grid) != len(threads):
            raise ValueError(f"grid and threads must have equal rank: {grid=}, {threads=}")
        global_size = tuple(int(g) * int(t) for g, t in zip(grid, threads))
        ndrange = remove_trailing_ones(global_size)

        # prepare launch parameters
        ndrange = (1,) if len(ndrange) == 0 else ndrange
        workgroupsize = remove_trailing_ones(threads)
        workgroupsize = (1,) if len(workgroupsize) == 0 else workgroupsize

        # run the kernel
        try:
            # the GPU timing events (CUDA, AMDGPU) are recorded in Julia directly around the launch
            self.launch_kernel_time, self.launch_kernel_time_start, start_evt, end_evt = self.launch_kernel(
                julia_func,
                args_tuple,
                params,
                ndrange,
                workgroupsize,
                int(self.smem_size),
                self.use_metal_buffer,
            )
            self._start_event = start_evt
            self._end_event = end_evt
        except JuliaError as e:
            if self.raise_errors:
                raise e
            else:
                raise SkippableFailure(f"Julia kernel launch failed for {params=}: {e}")

    def start_event(self):
        """Marks the start of a measurement (non-blocking).

        The GPU timing events (CUDA, AMDGPU) and Metal command buffers are recorded in Julia directly around the kernel
        launch in `run_kernel`, so that host-side work before the launch (e.g. Python to Julia calls, GC pauses) is
        not measured as GPU time.
        """
        self._host_start_time = perf_counter()
        self._start_event = None
        self._end_event = None

    def stop_event(self):
        """Marks the end of a measurement (non-blocking); the events are recorded in `run_kernel`."""
        pass

    def kernel_finished(self):
        """Returns True if the kernel has finished, False otherwise."""
        if self.backend_name == "CUDA":
            if self._end_event is not None:
                return bool(self.backend_mod.isdone(self._end_event))
            return True
        elif self.backend_mod_name == "AMDGPU":
            if self._end_event is not None:
                # AMDGPU.HIP.hipEventQuery returns (hipError_t, bool)
                result = self.backend_mod.HIP.hipEventQuery(self._end_event)
                if isinstance(result, tuple) and len(result) >= 2:
                    return bool(result[1])
                return bool(result)
            return True
        elif self.backend_mod_name == "Metal":
            return bool(self.metal_kernel_finished())
        elif self.backend_mod_name in ("INTEL", "CPU"):
            # No hardware events
            return True
        return True

    # @staticmethod
    def synchronize(self):
        try:
            self._host_time_julia = self.synchronize_gpu(self.launch_kernel_time_start)
            self._host_stop_time = perf_counter()
        except JuliaError as e:
            raise RuntimeError(f"Julia synchronize failed: {e}")

    # -------------------------
    # Memory utilities
    # -------------------------

    @staticmethod
    def memset(allocation, value, size):
        raise NotImplementedError("memset not yet implemented for Julia backend.")
        # try:
        #     jl.allocation_tmp = allocation
        #     jl.seval(f"CUDA.fill!(allocation_tmp, {int(value)})")
        #     del jl.allocation_tmp
        # except JuliaError as e:
        #     raise RuntimeError(f"Julia memset failed: {e}")

    @staticmethod
    def memcpy_dtoh(dest, src):
        """Perform a device to host memory copy."""
        try:
            if not isinstance(src, np.ndarray):
                # NumPy cannot view device memory, so on a (wrapped) GPU array `np.copyto` falls back to a very slow
                # generic conversion (seconds for a few million elements). Copy to a host `Array` in Julia instead,
                # which NumPy can then read through the buffer protocol without further copies.
                src = np.asarray(jl.Array(src))
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
        # self.smem_size = int(smem_args.get("size", 0))

    def copy_texture_memory_args(self, texmem_args):
        raise NotImplementedError(
            "Texture memory not yet supported in Julia backend. Submit a feature request if needed."
        )

    # -------------------------
    # Helper functions
    # -------------------------

    @staticmethod
    def package_installed(package):
        """Checks if the Julia package is a direct dependency of the environment, without loading it."""
        return bool(jl.seval(f'Base.find_package("{package}") !== nothing'))

    @staticmethod
    def package_available(package):
        """Checks if the Julia package can be imported."""
        try:
            jl.seval(f"import {package}")
            return True
        except Exception:
            return False

    def check_package_and_install(self, packages, allow_install=True, preferences=None):
        """Checks if the Julia package(s) are available, and installs the missing ones if not.

        ``preferences`` optionally maps package names to Julia preferences that are set before installing, e.g. to
        configure the CUDA toolkit JLLs. Packages that only appear in ``preferences`` are added as dependencies.
        """
        packages = [packages] if isinstance(packages, str) else list(packages)
        missing = [package for package in packages if not self.package_available(package)]
        if not missing:
            return
        add_command = "using Pkg; Pkg.add([" + ", ".join(f'\"{package}\"' for package in missing) + "])"
        if not allow_install:
            warn(f"{', '.join(missing)} not found in your Julia environment. Run `{add_command}` to install it.")
            return
        try:
            warn(f"{', '.join(missing)} not found, attempting to install it directly.")
            import juliapkg

            for package in missing:
                juliapkg.add(package)
            for package, package_preferences in (preferences or {}).items():
                # the preferences must be in place before installation, as CUDA.jl selects the toolkit at that point
                juliapkg.add(package, CUDA_TOOLKIT_JLLS.get(package), preferences=package_preferences)
            juliapkg.resolve()
            for package in missing:
                jl.seval(f"import {package}")
        except Exception as e:
            raise ImportError(
                f"{', '.join(missing)} not found in your Julia environment. Run `{add_command}` to install it."
            ) from e

    def setup_streams(self, backend_name: str):
        """Set up stream and event attributes for observers."""
        # Optional: common KernelAbstractions stream abstraction
        try:
            self.stream = self.backend_mod.get_default_stream()
        except Exception:
            self.stream = None

        # Set up stream and event attributes for observers
        if backend_name == "CUDA":
            self.stream = self.backend_mod.stream()
            self.start_evt = self.backend_mod.CuEvent
            self.end_evt = self.backend_mod.CuEvent
        elif backend_name == "AMD":
            self.stream = self.backend_mod.stream()
            self.start_evt = self.backend_mod.HIP.HIPEvent
            self.end_evt = self.backend_mod.HIP.HIPEvent
        elif backend_name == "INTEL":
            # OneAPI: no events available
            self.start_evt = None
            self.end_evt = None
        elif backend_name == "METAL":
            self.start_evt = self.start_event
            self.end_evt = self.stop_event
        elif backend_name == "CPU":
            # CPU, use host-side timing
            self.start_evt = None
            self.end_evt = None
        else:
            raise NotImplementedError(f"Backend {backend_name} not supported in Julia backend.")

    def process_compiler_options(self, compiler_options=None):
        """Process the given compiler options."""
        self.raise_errors = False
        for c in compiler_options or []:
            if c.lower().startswith("raise_errors="):
                raise_errors_str = c.split("=", 1)[1].strip().lower()
                if raise_errors_str in ("true", "1", "yes"):
                    self.raise_errors = True
                elif raise_errors_str in ("false", "0", "no"):
                    self.raise_errors = False
                else:
                    raise ValueError(f"Invalid value for raise_errors: {raise_errors_str}. Use true/false.")
                compiler_options.remove(c)

    def verify_backends_with_options(self, compiler_options=None):
        """Verify that the requested backend is available."""
        if compiler_options is not None and len(compiler_options) == 1:
            if compiler_options[0].upper() not in self.available_backends:
                raise ValueError(
                    f"Requested Julia backend '{compiler_options[0]}' not available. "
                    f"Available backends: {self.available_backends}"
                )
            backend_name = compiler_options[0].upper()
        else:
            if len(self.available_backends) != 1:
                if "CPU" in self.available_backends and len(self.available_backends) == 2:
                    self.available_backends.remove("CPU")
                else:
                    raise ValueError(
                        f"Multiple or no Julia backends detected: {self.available_backends}. "
                        "Please specify exactly one backend in compiler_options."
                    )
            backend_name = self.available_backends[0]
        return backend_name
