from kernel_tuner.observers.observer import BenchmarkObserver


class RegisterObserver(BenchmarkObserver):
    """Observer for counting the number of registers."""

    def get_results(self):
        try:
            registers_per_thread = self.dev.num_regs
        except AttributeError:
            raise NotImplementedError(
                f"Backend '{type(self.dev).__name__}' does not support count of registers per thread"
            )
        return {"num_regs": registers_per_thread}
