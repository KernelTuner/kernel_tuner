from kernel_tuner.observers.observer import BenchmarkObserver

class RegisterObserver(BenchmarkObserver):
    """Observer for counting the number of registers."""

    def __init__(self) -> None:
        super().__init__()

    def get_results(self):
        return {
            "num_regs": self.dev.num_regs
        }