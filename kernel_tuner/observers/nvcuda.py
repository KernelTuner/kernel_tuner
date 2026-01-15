import numpy as np

# Support both cuda-python < 13 and >= 13 import structures
try:
    # cuda-python >= 13 uses cuda.bindings module
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        # cuda-python < 13 uses direct imports
        from cuda import cudart
    except ImportError:
        cudart = None

from kernel_tuner.observers.observer import BenchmarkObserver
from kernel_tuner.util import cuda_error_check


class CudaRuntimeObserver(BenchmarkObserver):
    """Observer that measures time using CUDA events during benchmarking"""

    def __init__(self, dev):
        self.dev = dev
        self.stream = dev.stream
        self.start = dev.start
        self.end = dev.end
        self.times = []

    def after_finish(self):
        # Time is measured in milliseconds
        err, time = cudart.cudaEventElapsedTime(self.start, self.end)
        cuda_error_check(err)
        self.times.append(time)

    def get_results(self):
        results = {"time": np.average(self.times), "times": self.times.copy()}
        self.times = []
        return results
