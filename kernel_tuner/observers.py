

from abc import ABC, abstractmethod

class BenchmarkObserver(ABC):
    """Base class for Benchmark Observers"""

    def before_start(self):
        pass

    def after_start(self):
        pass

    def during(self):
        pass

    def after_finish(self):
        pass

    @abstractmethod
    def get_results(self):
        pass


