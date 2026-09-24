from time import perf_counter
from warnings import warn

import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver, PrologueObserver


class JuliaRuntimeObserver(BenchmarkObserver):
    """Cross-backend GPU timing for KernelAbstractions.

    - CUDA: CuEvent timing
    - ROCBackend: HIPEvent timing
    - OneAPI: host-side timing (less accurate, no events available)
    - Metal: timing by wrapping the kernel launch with a command buffer
    """

    def __init__(
        self,
        kernelabstractions,
        kt_backend,
        jl_backend,
        jl_backend_mod,
        jl_backend_name,
        stream=None,
        start_event=None,
        end_event=None,
    ):
        """Observer that measures GPU time depending on the Julia backend used."""
        self.kernelabstractions = kernelabstractions
        self.kt_backend = kt_backend
        self.backend = jl_backend
        self.backend_mod = jl_backend_mod
        self.name = jl_backend_name.lower()
        self.stream = stream
        self.start = start_event
        self.end = end_event
        self.times = []

    def before_start(self):
        # Events are now recorded in the backend's start_event() method
        pass

    def after_finish(self):
        ms = None
        ms_hostside = self.kt_backend._host_time_julia
        ms_overhead = 0.0

        if self.name == "cuda":
            # CUDA: use events stored in backend
            if self.kt_backend._start_event is not None and self.kt_backend._end_event is not None:
                ms = float(self.backend_mod.elapsed(self.kt_backend._start_event, self.kt_backend._end_event) * 1000.0)
        elif self.name == "amdgpu":
            # AMD: use events stored in backend
            if self.kt_backend._start_event is not None and self.kt_backend._end_event is not None:
                ms = float(self.backend_mod.HIP.elapsed(self.kt_backend._start_event, self.kt_backend._end_event) * 1000.0)
        elif self.name == "metal":
            # register the overhead from the command buffer 
            buf = self.kt_backend.metal_get_global_buffer()
            if buf is not None:
                ms_overhead = float((buf.GPUEndTime - buf.GPUStartTime) * 1000.0)
        elif self.name in ("intel", "cpu"):
            # uses host-side timing
            pass

        # If GPU timing failed or not available, fall back to host-side timing if available
        if ms is None and ms_hostside is not None:
            ms = ms_hostside
            if self.name == "cuda" or self.name == "amdgpu":
                warn(f"Using host-side timing for Julia {self.name} backend; results may be less accurate.")
        if ms is None:
            if self.kt_backend._host_start_time is not None and self.kt_backend._host_stop_time is not None:
                ms_hostside_python = (self.kt_backend._host_stop_time - self.kt_backend._host_start_time) * 1000.0
                ms = ms_hostside_python
                warn(f"Using Python host-side timing for Julia {self.name} backend; results may be less accurate.")
        if ms is None:
            raise RuntimeError(f"Failed to measure GPU time for Julia {self.name} backend; no timing information available.")
        else:
            # Subtract any known overhead
            ms -= ms_overhead

        # If both GPU and host timing are available, check for discrepancies
        if ms is not None and ms_hostside is not None and ms > ms_hostside and ms_hostside > 0.0:
            if ms < 1 and (ms > 1.2 * ms_hostside):
                warn(
                    f"Measured GPU time {ms:.3f} ms is greater than host time {ms_hostside:.3f} ms; "
                    "this may happen with very short execution times."
                )
            elif ms > 1.2 * ms_hostside:
                warn(
                    f"Measured GPU time {ms:.3f} ms is substantially greater than host time {ms_hostside:.3f} ms; "
                    "this may indicate an issue with the timing measurement."
                )

        self.times.append(ms)

    def get_results(self):
        times_avg = self.times.copy()
        if len(self.times) >= 5:
            # remove the first to avoid JIT warmup effects
            times_avg = self.times[1:]
        results = {
            "time": np.average(times_avg),
            "times": self.times.copy(),
        }
        self.times = []
        return results


class JuliaJITWarmup(PrologueObserver):
    """Prologue observer to enforce warmup before every configuration to trigger JIT."""

    def __init__(self, backend):
        """Not implemented, just to trigger JIT."""
        pass

    def before_start(self):
        """Not implemented, just to trigger JIT."""
        pass

    def after_finish(self):
        """Not implemented, just to trigger JIT."""
        pass

    def get_results(self):
        return {}
