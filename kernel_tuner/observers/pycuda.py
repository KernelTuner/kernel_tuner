import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver


class PyCudaRuntimeObserver(BenchmarkObserver):
    """Observer that measures time using CUDA events during benchmarking"""

    def __init__(self, dev):
        self.dev = dev
        self.stream = dev.stream
        self.start = dev.start
        self.end = dev.end
        self.times = []

    def after_finish(self):
        # Time is measured in milliseconds
        self.times.append(self.end.time_since(self.start))

    def get_results(self):
        results = {"time": np.average(self.times), "times": self.times.copy()}
        self.times = []
        return results
