""" The default strategy that iterates through the whole parameter space """
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common

_options = {}

def tune(runner, tuning_options):

    # create the searchspace
    searchspace = Searchspace(tuning_options, runner.dev.max_threads, sort=True)

    # call the runner
    results, env = runner.run(searchspace.list, tuning_options)

    return results, env


tune.__doc__ = common.get_strategy_docstring("Brute Force", _options)
