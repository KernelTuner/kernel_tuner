from time import perf_counter
from warnings import warn
import logging

import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver, PrologueObserver

logger = logging.getLogger(__name__)


class JuliaRuntimeObserver(BenchmarkObserver):
    """Cross-backend GPU timing for KernelAbstractions.

    - CUDA: CuEvent timing
    - ROCBackend: HIPEvent timing
    - OneAPI: host-side timing (less accurate, no events available)
    - Metal: GPU start and end times of the command buffers that ran the kernel
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

        if self.name == "cuda":
            # CUDA: use events stored in backend
            if self.kt_backend._start_event is not None and self.kt_backend._end_event is not None:
                ms = float(self.backend_mod.elapsed(self.kt_backend._start_event, self.kt_backend._end_event) * 1000.0)
        elif self.name == "amdgpu":
            # AMD: use events stored in backend
            if self.kt_backend._start_event is not None and self.kt_backend._end_event is not None:
                ms = float(self.backend_mod.HIP.elapsed(self.kt_backend._start_event, self.kt_backend._end_event) * 1000.0)
        elif self.name == "metal":
            # GPU time of the command buffers that ran the kernel
            gpu_ms = self.kt_backend.metal_gpu_time_ms()
            if gpu_ms is not None:
                ms = float(gpu_ms)
        elif self.name in ("intel", "cpu"):
            # uses host-side timing
            pass

        # If GPU timing failed or not available, fall back to host-side timing if available
        if ms is None and ms_hostside is not None:
            ms = ms_hostside
            if self.name in ("cuda", "amdgpu", "metal"):
                warn(f"Using host-side timing for Julia {self.name} backend; results may be less accurate.")
        if ms is None:
            if self.kt_backend._host_start_time is not None and self.kt_backend._host_stop_time is not None:
                ms_hostside_python = (self.kt_backend._host_stop_time - self.kt_backend._host_start_time) * 1000.0
                ms = ms_hostside_python
                warn(f"Using Python host-side timing for Julia {self.name} backend; results may be less accurate.")
        if ms is None:
            raise RuntimeError(f"Failed to measure GPU time for Julia {self.name} backend; no timing information available.")

        # If both GPU and host timing are available, check for discrepancies
        if ms is not None and ms_hostside is not None and ms > ms_hostside and ms_hostside > 0.0:
            if ms > 10 * ms_hostside:
                warn(
                    f"Measured GPU time {ms:.3f} ms is more than 10x host time {ms_hostside:.3f} ms; "
                    "this may indicate an issue with the timing measurement. Replacing GPU time with host time for this measurement."
                )
                ms = ms_hostside
            elif ms < 1 and (ms > 1.2 * ms_hostside):
                logger.info(
                    f"Measured GPU time {ms:.3f} ms is greater than host time {ms_hostside:.3f} ms; "
                    "this may happen with very short execution times."
                )
            elif ms > 1.2 * ms_hostside:
                warn(
                    f"Measured GPU time {ms:.3f} ms is substantially greater than host time {ms_hostside:.3f} ms; "
                    "this may indicate an issue with the timing measurement."
                )

        self.times.append(ms)

    # a measured time is an outlier if it deviates from the median both statistically (robust z-score, based on the
    # median absolute deviation) and substantially (relative to the median), so that tiny differences between otherwise
    # consistent measurements (e.g. 0.031 vs. 0.033 ms) are not regarded as outliers
    outlier_robust_z = 3.0
    outlier_rel_deviation = 0.25
    outlier_warn_fraction = 0.1

    @classmethod
    def find_outliers(cls, arr):
        """Returns a boolean mask of the times in `arr` that are outliers (only slower times are regarded as such)."""
        median = np.median(arr)
        deviation = arr - median
        substantial = deviation > cls.outlier_rel_deviation * median
        mad = np.median(np.abs(deviation))
        if mad == 0:
            # at least half of the times are identical, any substantial deviation is an outlier
            return substantial
        robust_z = 0.6745 * deviation / mad
        return substantial & (robust_z > cls.outlier_robust_z)

    def get_results(self):
        times_avg = self.times.copy()
        if len(self.times) >= 5:
            # due to interactivity, the Julia runtime observer may have rare outliers; remove them if there are any
            arr = np.array(times_avg)
            outliers = self.find_outliers(arr)
            num_outliers = int(np.sum(outliers))
            if num_outliers > 0:
                log_str = (
                    f"Removed {num_outliers} of {len(arr)} measured times as outliers "
                    f"({', '.join(f'{t:.4f}' for t in arr[outliers])} ms vs. median {np.median(arr):.4f} ms)."
                )
                # only warn when outliers are frequent, as that indicates unreliable measurements
                if num_outliers > self.outlier_warn_fraction * len(arr):
                    logger.warning(log_str)
                else:
                    logger.info(log_str)
                times_avg = arr[~outliers]
            else:
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
