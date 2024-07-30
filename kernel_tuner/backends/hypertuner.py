"""This module contains a 'device' for hyperparameter tuning using the autotuning methodology."""

from pathlib import Path

from kernel_tuner.backends.backend import Backend

try:
    methodology_available = True
    from autotuning_methodology.report_experiments import get_strategy_scores
except ImportError:
    methodology_available = False

class HypertunerFunctions(Backend):
    """Class for executing hyperparameter tuning."""

    def __init__(self, iterations):
        self.iterations = iterations
        if methodology_available is not True:
            raise ImportError("Unable to import the autotuning methodology, run `pip install autotuning_methodology`.")

    def ready_argument_list(self, arguments):
        return super().ready_argument_list(arguments)
    
    def compile(self, kernel_instance):
        return super().compile(kernel_instance)
    
    def start_event(self):
        return super().start_event()
    
    def stop_event(self):
        return super().stop_event()
    
    def kernel_finished(self):
        super().kernel_finished()
        return True
    
    def synchronize(self):
        return super().synchronize()
    
    def run_kernel(self, func, gpu_args=None, threads=None, grid=None, stream=None):
        # generate the experiments file
        experiments_filepath = Path('.')

        # run the methodology to get a fitness score for this configuration
        scores = get_strategy_scores(str(experiments_filepath))
        score = scores[list(scores.keys()[0])]['score']
        return score
    
    def memset(self, allocation, value, size):
        return super().memset(allocation, value, size)
    
    def memcpy_dtoh(self, dest, src):
        return super().memcpy_dtoh(dest, src)
    
    def memcpy_htod(self, dest, src):
        return super().memcpy_htod(dest, src)
