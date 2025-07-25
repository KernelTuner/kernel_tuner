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
    """BenchmarkObserver subclass for registering the hyperparameter tuning score."""

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

    def __init__(self, iterations, compiler_options=None):
        self.iterations = iterations
        self.compiler_options = compiler_options
        self.observers = [ScoreObserver(self)]
        self.name = platform.processor()
        self.max_threads = 1024
        self.last_score = None

        # set the defaults
        self.gpus = ["A100", "A4000", "MI250X"]
        folder = "../autotuning_methodology/benchmark_hub/kernels"
        self.applications = [
            {
                "name": "dedispersion_milo",
                "folder": folder,
                "input_file": "dedispersion_milo.json",
                "objective_performance_keys": ["time"]
            },
            {
                "name": "hotspot_milo",
                "folder": folder,
                "input_file": "hotspot_milo.json",
                "objective_performance_keys": ["GFLOP/s"]
            },
            {
                "name": "convolution_milo",
                "folder": folder,
                "input_file": "convolution_milo.json",
                "objective_performance_keys": ["time"]
            },
            {
                "name": "gemm_milo",
                "folder": folder,
                "input_file": "gemm_milo.json",
                "objective_performance_keys": ["time"]
            }
        ]
        # any additional settings
        self.override = { 
            "experimental_groups_defaults": { 
                "repeats": 25,
                "samples": self.iterations,
                "minimum_fraction_of_budget_valid": 0.1,
                "minimum_number_of_valid_search_iterations": 5,
            },
            "statistics_settings": {
                "cutoff_percentile": 0.95,
                "cutoff_percentile_start": 0.01,
                "cutoff_type": "time",
                "objective_time_keys": [
                    "all"
                ]
            }
        }

        # override the defaults with compiler options if provided
        if self.compiler_options is not None:
            if "gpus" in self.compiler_options:
                self.gpus = self.compiler_options["gpus"]
            if "applications" in self.compiler_options:
                self.applications = self.compiler_options["applications"]
            if "override" in self.compiler_options:
                self.override = self.compiler_options["override"]

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

        name = kernel_instance.name if len(kernel_instance.name) > 0 else kernel_instance.kernel_source.kernel_name
        experiments_filepath = generate_experiment_file(name, path, searchspace_strategies, self.applications, self.gpus, 
                                                        override=self.override, generate_unique_file=True, overwrite_existing_file=True)
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
        # from cProfile import Profile
    
        # # generate the experiments file
        # experiments_filepath = Path(func)

        # # run the methodology to get a fitness score for this configuration
        # with Profile() as pr:
        #     scores = get_strategy_scores(str(experiments_filepath), full_validate_on_load=False)
        #     pr.dump_stats('diff_evo_hypertune_hotspot.prof')
        # self.last_score = scores[list(scores.keys())[0]]['score']
        # raise ValueError(scores)
    
        # generate the experiments file
        experiments_filepath = Path(func)

        # run the methodology to get a fitness score for this configuration
        scores = get_strategy_scores(str(experiments_filepath), full_validate_on_load=False)
        self.last_score = scores[list(scores.keys())[0]]['score']

        # remove the experiments file
        experiments_filepath.unlink()
    
    def memset(self, allocation, value, size):
        return super().memset(allocation, value, size)
    
    def memcpy_dtoh(self, dest, src):
        return super().memcpy_dtoh(dest, src)
    
    def memcpy_htod(self, dest, src):
        return super().memcpy_htod(dest, src)

    def refresh_memory(self, device_memory, host_arguments, should_sync):
        """This is a no-op for the hypertuner backend, as it does not manage memory directly."""
        pass
