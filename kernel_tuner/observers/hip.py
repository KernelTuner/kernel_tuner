import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver

try:
    from pyhip import hip, hiprtc
except ImportError:
    print("Not able to import pyhip, check if PYTHONPATH includes PyHIP")
    hip = None
    hiprtc = None


class HipRuntimeObserver(BenchmarkObserver):
    """Observer that measures time using CUDA events during benchmarking"""

    def __init__(self, dev):
        self.dev = dev
        self.stream = dev.stream
        self.start = dev.start
        self.end = dev.end
        self.times = []

    def after_finish(self):
        # Time is measured in milliseconds
        EventElapsedTime = hip.hipEventElapsedTime(self.start, self.end)
        self.times.append(EventElapsedTime.value)

    def get_results(self):
        results = {"time": np.average(self.times), "times": self.times.copy()}
        self.times = []
        return results
