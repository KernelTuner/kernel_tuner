import ray

from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner.runners.simulation import SimulationRunner

@ray.remote
class RemoteActor():
    def __init__(self, 
                 kernel_source,
                 kernel_options, 
                 device_options,
                 iterations,
                 observers,
                 cache_manager=None,
                 simulation_mode=False):
        self.kernel_source = kernel_source
        self.kernel_options = kernel_options
        self.device_options = device_options
        self.iterations = iterations
        self.observers = observers
        self.cache_manager = cache_manager
        self.simulation_mode = simulation_mode
        self.runner = None
            
    def execute(self, tuning_options, strategy=None, searchspace=None, element=None):
        if self.runner is None:
            self.init_runner()
        if strategy and searchspace:
            results = strategy.tune(searchspace, self.runner, tuning_options)
            return results, tuning_options
        elif element:
            return self.runner.run([element], tuning_options)[0]
        else:
            raise ValueError("Invalid arguments for ray actor's execute method.")
        
    def set_cache_manager(self, cache_manager):
        if self.cache_manager is None:
            self.cache_manager = cache_manager

    def get_cache_magaer(self):
        return self.cache_manager
    
    def init_runner(self):
        if self.cache_manager is None:
            raise ValueError("Cache manager is not set.")
        if self.simulation_mode:
            self.runner = SimulationRunner(self.kernel_source, self.kernel_options, self.device_options, 
                                            self.iterations, self.observers)
        else:
            self.runner = SequentialRunner(self.kernel_source, self.kernel_options, self.device_options, 
                                       self.iterations, self.observers, cache_manager=self.cache_manager)
