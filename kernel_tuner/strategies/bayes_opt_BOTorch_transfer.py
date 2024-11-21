"""Bayesian Optimization implementation using BO Torch."""

try:
    from torch import Tensor
    bayes_opt_present = True
except ImportError:
    bayes_opt_present = False

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.bayes_opt_BOTorch import BayesianOptimization


def tune(searchspace: Searchspace, runner, tuning_options):
    """The entry function for tuning a searchspace using this algorithm."""
    max_fevals = tuning_options.strategy_options.get("max_fevals", 100)
    bo = BayesianOptimization(searchspace, runner, tuning_options)
    return bo.run(max_fevals)

class BayesianOptimizationTransfer(BayesianOptimization):
    """Bayesian Optimization class with transfer learning."""

    def __init__(self, searchspace: Searchspace, runner, tuning_options):
        super().__init__(searchspace, runner, tuning_options)

    def run_config(self, config: tuple):
        return super().run_config(config)
    
    def evaluate_configs(self, X: Tensor):
        return super().evaluate_configs(X)
    
    def initial_sample(self):
        return super().initial_sample()
    
    def initialize_model(self, state_dict=None, exact=True):
        return super().initialize_model(state_dict, exact)
    
    def run(self, max_fevals: int, max_batch_size=2048):
        return super().run(max_fevals, max_batch_size)