import ray
import sys
import copy

from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner.runners.simulation import SimulationRunner
from kernel_tuner.core import DeviceInterface
from kernel_tuner.observers.register import RegisterObserver
from kernel_tuner.util import get_gpu_id, get_gpu_type

@ray.remote
class RemoteActor():
    def __init__(self, 
                 kernel_source,
                 kernel_options, 
                 device_options,
                 iterations,
                 observers_type_and_arguments,
                 id,
                 cache_manager=None,
                 simulation_mode=False):
        self.kernel_source = kernel_source
        self.kernel_options = kernel_options
        self.device_options = device_options
        self.iterations = iterations
        self.cache_manager = cache_manager
        self.simulation_mode = simulation_mode
        self.runner = None
        self.id = get_gpu_id(kernel_source.lang) if not simulation_mode else None
        self._reinitialize_observers(observers_type_and_arguments)
        

    def execute(self, tuning_options, strategy=None, searchspace=None, element=None):
        tuning_options['observers'] = self.observers
        if self.runner is None:
            self.init_runner()
        if strategy and searchspace:
            results = strategy.tune(searchspace, self.runner,  tuning_options)
            # observers can't be pickled
            tuning_options['observers'] = None
            return results, tuning_options
        elif element:
            results = self.runner.run([element],  tuning_options)[0]
            # observers can't be pickled
            tuning_options['observers'] = None
            return results, tuning_options
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

    def _reinitialize_observers(self, observers_type_and_arguments):
        # observers can't be pickled to the actor so we need to re-initialize them
        register_observer = False
        self.observers = []
        for (observer, arguments) in observers_type_and_arguments:
            if "device" in arguments:
                arguments["device"] = self.id
            if isinstance(observer, RegisterObserver):
                register_observer = True
            else:
                self.observers.append(observer(**arguments))
        # we dont initialize the dev with observers, as this creates a 'invalid resource handle' error down the line
        self.dev = DeviceInterface(self.kernel_source, iterations=self.iterations, **self.device_options) if not self.simulation_mode else None
        # the register observer needs dev to be initialized, that's why its done later
        if register_observer:
            self.observers.append(RegisterObserver(self.dev))

    def get_gpu_type(self, lang):
        print(f"DEBUG:actor get_gpu_type called", file=sys.stderr)
        return get_gpu_type(lang)
