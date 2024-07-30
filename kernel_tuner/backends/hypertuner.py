"""This module contains a 'device' for hyperparameter tuning using the autotuning methodology."""

import platform
from pathlib import Path

from kernel_tuner.backends.backend import Backend

try:
    methodology_available = True
    from autotuning_methodology.report_experiments import get_strategy_scores
except ImportError:
    methodology_available = False

class HypertunerFunctions(Backend):
    """Class for executing hyperparameter tuning."""
    units = {}

    def __init__(self, iterations):
        self.iterations = iterations
        self.observers = []
        self.name = platform.processor()
        self.max_threads = 1024

        # set the environment options
        env = dict()
        env["iterations"] = self.iterations
        self.env = env

        # check for the methodology package
        if methodology_available is not True:
            raise ImportError("Unable to import the autotuning methodology, run `pip install autotuning_methodology`.")

    def ready_argument_list(self, arguments):
        arglist = super().ready_argument_list(arguments)
        if arglist is None:
            arglist = []
        return arglist
    
    def compile(self, kernel_instance):
        super().compile(kernel_instance)
        # TODO implement experiments file generation
        params = kernel_instance.params
        raise NotImplementedError(f'Not yet implemented: experiments generation for {params}')
        experiments_filepath = Path('.')
        return str(experiments_filepath)
    
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
        experiments_filepath = Path(func)

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
