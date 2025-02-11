"""This module contains a 'device' for hyperparameter tuning using the autotuning methodology."""

import platform
from pathlib import Path

from numpy import mean

from kernel_tuner.backends.backend import Backend
from kernel_tuner.observers.observer import BenchmarkObserver

try:
    methodology_available = True
    from autotuning_methodology.experiments import generate_experiment_file
    from autotuning_methodology.report_experiments import get_strategy_scores
except ImportError:
    methodology_available = False


class ScoreObserver(BenchmarkObserver):
    def __init__(self, dev):
        self.dev = dev
        self.scores = []

    def after_finish(self):
        self.scores.append(self.dev.last_score)

    def get_results(self):
        results = {'score': mean(self.scores), 'scores': self.scores.copy()}
        self.scores = []
        return results

class HypertunerFunctions(Backend):
    """Class for executing hyperparameter tuning."""
    units = {}

    def __init__(self, iterations):
        self.iterations = iterations
        self.observers = [ScoreObserver(self)]
        self.name = platform.processor()
        self.max_threads = 1024
        self.last_score = None

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
        path = Path(__file__).parent.parent.parent / "hyperparamtuning"
        path.mkdir(exist_ok=True)

        # TODO get applications & GPUs args from benchmark
        gpus = ["RTX_3090", "RTX_2080_Ti"]
        applications = None
        # applications = [
        #     {
        #         "name": "convolution",
        #         "folder": "./cached_data_used/kernels",
        #         "input_file": "convolution.json"
        #     },
        #     {
        #         "name": "pnpoly",
        #         "folder": "./cached_data_used/kernels",
        #         "input_file": "pnpoly.json"
        #     }
        # ]

        # strategy settings
        strategy: str = kernel_instance.arguments[0]
        hyperparams = [{'name': k, 'value': v} for k, v in kernel_instance.params.items()]
        hyperparams_string = "_".join(f"{k}={str(v)}" for k, v in kernel_instance.params.items())
        searchspace_strategies = [{
            "autotuner": "KernelTuner",
            "name": f"{strategy.lower()}_{hyperparams_string}",
            "display_name": strategy.replace('_', ' ').capitalize(),
            "search_method": strategy.lower(),
            'search_method_hyperparameters': hyperparams
        }]

        # any additional settings
        override = { 
            "experimental_groups_defaults": { 
                "samples": self.iterations 
            }
        }

        name = kernel_instance.name if len(kernel_instance.name) > 0 else kernel_instance.kernel_source.kernel_name
        experiments_filepath = generate_experiment_file(name, path, searchspace_strategies, applications, gpus, 
                                                        override=override, overwrite_existing_file=True)
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
        self.last_score = scores[list(scores.keys())[0]]['score']
    
    def memset(self, allocation, value, size):
        return super().memset(allocation, value, size)
    
    def memcpy_dtoh(self, dest, src):
        return super().memcpy_dtoh(dest, src)
    
    def memcpy_htod(self, dest, src):
        return super().memcpy_htod(dest, src)
