import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from kernel_tuner.observers.observer import BenchmarkObserver


class CupyRuntimeObserver(BenchmarkObserver):
    """Observer that measures time using CUDA events during benchmarking in the CuPy backend"""

    def __init__(self, dev):
        self.dev = dev
        self.stream = dev.stream
        self.start = dev.start
        self.end = dev.end
        self.times = []

    def after_finish(self):
        # Time is measured in milliseconds
        self.times.append(cp.cuda.get_elapsed_time(self.start, self.end))

    def get_results(self):
        results = {"time": np.average(self.times), "times": self.times.copy()}
        self.times = []
        return results
