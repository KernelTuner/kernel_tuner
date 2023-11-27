import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver


class CompilerRuntimeObserver(BenchmarkObserver):
    """Observer that collects results returned by benchmarking function in the C backend"""

    def __init__(self, dev):
        self.dev = dev
        self.objective = "time"
        self.times = []

    def after_finish(self):
        self.times.append(self.dev.last_result)

    def get_results(self):
        results = {
            self.objective: np.average(self.times),
            self.objective + "s": self.times.copy(),
        }
        self.times = []
        return results
