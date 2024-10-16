"""This module contains a 'device' for hyperparameter tuning using the autotuning methodology."""

import platform
from pathlib import Path

from kernel_tuner.backends.backend import Backend

try:
    methodology_available = True
    from autotuning_methodology.experiments import generate_experiment_file
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
        # params = kernel_instance.params
        path = Path(__file__).parent

        # TODO: get applications & GPUs args from benchmark
        applications = ['convolution', 'pnpoly']
        gpus = ["RTX_3090", "RTX_2080_Ti"]

        strategy = kernel_instance.arguments[0]
        searchspace_strategies = [{
            "autotuner": "KernelTuner",
            "name": strategy,
            "display_name": strategy,
            "search_method": strategy,
            'search_method_hyperparameters': kernel_instance.params
        }]

        override = { 
            "experimental_groups_defaults": { 
                "samples": kernel_instance.iterations 
            } 
        }

        experiments_filepath = generate_experiment_file(kernel_instance.name, path, applications, gpus, searchspace_strategies, override=override)
        return str(experiments_filepath)
        return lambda e: "not implemented"  # the compile function is expected to return a function
        # raise NotImplementedError(f'Not yet implemented: experiments generation for {params}')
    
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
        raise ValueError(func, gpu_args)
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
