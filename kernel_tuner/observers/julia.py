import numpy as np
from kernel_tuner.observers.observer import BenchmarkObserver, PrologueObserver

class JuliaRuntimeObserver(BenchmarkObserver):
    """Observer that measures GPU time using CUDA.CuEvent and CUDA.elapsed."""

    def __init__(self, CUDA):
        self.CUDA = CUDA
        self.start = self.CUDA.CuEvent()
        self.end   = self.CUDA.CuEvent()
        self.stream = self.CUDA.stream()   # default stream
        self.times = []

    def before_start(self):
        # record start event
        self.CUDA.record(self.start, self.stream)

    def after_finish(self):
        # record end event
        self.CUDA.record(self.end, self.stream)
        self.CUDA.synchronize(self.end)

        # milliseconds
        ms = float(self.CUDA.elapsed(self.start, self.end))
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

    def __init__(self, CUDA):
        pass

    def before_start(self):
        pass

    def after_finish(self):
        pass

    def get_results(self):
        return {}
