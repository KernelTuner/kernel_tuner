import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver

try:
    from pyhip import hip, hiprtc
except ImportError:
    hip = None
    hiprtc = None


class HipRuntimeObserver(BenchmarkObserver):
    """Observer that measures time using CUDA events during benchmarking."""

    def __init__(self, dev):
        if not hip or not hiprtc:
            raise ImportError("Unable to import PyHIP, make sure PYTHONPATH includes PyHIP, or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#hip-and-pyhip.")

        self.dev = dev
        self.stream = dev.stream
        self.start = dev.start
        self.end = dev.end
        self.times = []

    def after_finish(self):
        # Time is measured in milliseconds
        EventElapsedTime = hip.hipEventElapsedTime(self.start, self.end)
        self.times.append(EventElapsedTime)

    def get_results(self):
        results = {"time": np.average(self.times), "times": self.times.copy()}
        self.times = []
        return results
