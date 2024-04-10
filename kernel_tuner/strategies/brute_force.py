""" The default strategy that iterates through the whole parameter space """
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.runners.parallel import ParallelRunner
from kernel_tuner.runners.ray.cache_manager import CacheManager

_options = {}

def tune(searchspace: Searchspace, runner, tuning_options):

    if isinstance(runner, ParallelRunner):
        cache_manager = CacheManager.remote(tuning_options)
        return runner.run(searchspace.sorted_list(), tuning_options, cache_manager)
    else:
        return runner.run(searchspace.sorted_list(), tuning_options)


tune.__doc__ = common.get_strategy_docstring("Brute Force", _options)
