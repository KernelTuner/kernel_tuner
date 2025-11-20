import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver

try:
    import torch
except (ImportError, RuntimeError):
    torch = None


class TritonRuntimeObserver(BenchmarkObserver):
    """Observer that measures time using CUDA events during benchmarking."""

    def __init__(self, dev):
        if torch is None:
            raise ImportError("Unable to load torch")

        self.dev = dev
        self.stream = dev.stream
        self.start = dev.start
        self.end = dev.end
        self.times = []

    def after_finish(self):
        # Time is measured in milliseconds
        event_elapsed_time = self.start.elapsed_time(self.end)
        self.times.append(event_elapsed_time)

    def get_results(self):
        results = {"time": np.average(self.times), "times": self.times.copy()}
        self.times = []
        return results