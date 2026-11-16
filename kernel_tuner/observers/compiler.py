import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver


class CompilerDefaultObserver(BenchmarkObserver):
    """Observer that collects results returned by benchmarking function in the C backend"""

    def __init__(self, dev):
        self.dev = dev
        # The default objective name if a single float is returned
        self.objective = "time"
        self.results = {}

    def after_finish(self):
        # Result can be either a single float, or a dict of str:float keyvals

        if not isinstance(self.dev.last_result, dict):
            # Single float result is treated as self.objective
            last_result = { self.objective: self.dev.last_result }
        else:
            last_result = self.dev.last_result.copy()

        for obs in last_result:
            plural = obs + "s"
            if plural not in self.results:
                self.results[plural] = []
            self.results[plural].append(last_result[obs])

    def get_results(self):
        # Return both the lists of individual results and their averages
        results = self.results | { obs[:-1]: np.average(self.results[obs]) for obs in self.results }

        self.results = {}

        return results
