from time import perf_counter
from warnings import warn

import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver, PrologueObserver


class JuliaRuntimeObserver(BenchmarkObserver):
    """Cross-backend GPU timing for KernelAbstractions.

    - CUDA: CuEvent timing
    - ROCBackend: ROCEvent timing
    - OneAPI: host timing + synchronize (less accurate, no events available)
    - Metal: host timing + synchronize
    """

    def __init__(
        self,
        kernelabstractions,
        backend,
        backend_mod,
        backend_name,
        stream=None,
        start_event=None,
        end_event=None,
    ):
        """Observer that measures GPU time depending on the Julia backend used."""
        self.kernelabstractions = kernelabstractions
        self.backend = backend
        self.backend_mod = backend_mod
        self.name = backend_name.lower()
        self.stream = stream
        self.start = start_event
        self.end = end_event
        self.times = []
        self.t0 = None

        if self.name == "cuda":
            # initialize events for this instance of the observer
            self.start = self.start()
            self.end = self.end()
            self.stream = backend_mod.stream()
        elif self.name == "amdgpu":
            self.stream = backend_mod.default_stream()
            self.start = self.start(self.stream, timing=True)
            self.end = self.end(self.stream, timing=True)

    def before_start(self):
        if self.start is not None:
            if self.name == "metal":
                self.t0 = self.start()
            elif self.name == "amdgpu":
                self.backend_mod.record(self.start)
            else:
                self.backend_mod.record(self.start, self.stream)
        else:
            # fallback: host-side timestamp
            self.t0 = perf_counter()

    def after_finish(self):
        if self.end is not None:
            if self.name == "metal":
                ms = float((self.end() - self.t0) * 1000.0)
            elif self.name == "amdgpu":
                self.backend_mod.record(self.end)
                ms = float(self.backend_mod.elapsed(self.start, self.end) * 1000.0)
            else:
                self.backend_mod.synchronize(self.end)
                self.backend_mod.record(self.end, self.stream)
                self.backend_mod.synchronize(self.end)
                ms = float(self.backend_mod.elapsed(self.start, self.end))
        else:
            self.kernelabstractions.synchronize(self.backend)
            dt = perf_counter() - self.t0
            ms = dt * 1000.0
            warn(f"Using host-side timing for Julia {self.name} backend; results may be less accurate.")

        self.times.append(ms)

    def get_results(self):
        results = {
            "time": np.average(self.times),
            "times": self.times.copy(),
        }
        self.times = []
        return results


class JuliaJITWarmup(PrologueObserver):
    """Prologue observer to enforce warmup before every configuration to trigger JIT."""

    def __init__(self, backend):
        pass

    def before_start(self):
        pass

    def after_finish(self):
        pass

    def get_results(self):
        return {}
