import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver


class OpenCLObserver(BenchmarkObserver):
    """Observer that measures time using CUDA events during benchmarking"""

    def __init__(self, dev):
        self.dev = dev
        self.times = []

    def after_finish(self):
        event = self.dev.event
        # Time is converted to milliseconds
        self.times.append((event.profile.end - event.profile.start) * 1e-6)

    def get_results(self):
        results = {"time": np.average(self.times), "times": self.times.copy()}
        self.times = []
        return results
