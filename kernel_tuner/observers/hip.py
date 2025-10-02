import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver

try:
    from hip import hip, hiprtc
except (ImportError, RuntimeError):
    hip = None
    hiprtc = None


def hip_check(call_result):
    """helper function to check return values of hip calls"""
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        _, error_name = hip.hipGetErrorName(err)
        _, error_str = hip.hipGetErrorString(err)
        raise RuntimeError(f"{error_name}: {error_str}")
    return result


class HipRuntimeObserver(BenchmarkObserver):
    """Observer that measures time using CUDA events during benchmarking."""

    def __init__(self, dev):
        if not hip or not hiprtc:
            raise ImportError("Unable to import HIP Python, or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#hip-and-hip-python.")

        self.dev = dev
        self.stream = dev.stream
        self.start = dev.start
        self.end = dev.end
        self.times = []

    def after_finish(self):
        # Time is measured in milliseconds
        EventElapsedTime = hip_check(hip.hipEventElapsedTime(self.start, self.end))
        self.times.append(EventElapsedTime)

    def get_results(self):
        results = {"time": np.average(self.times), "times": self.times.copy()}
        self.times = []
        return results
