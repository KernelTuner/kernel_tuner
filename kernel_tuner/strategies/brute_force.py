"""The default strategy that iterates through the whole parameter space."""
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common

_options = {}

def tune(searchspace: Searchspace, runner, tuning_options):

    # Force error on unsupported options
    common.get_options(tuning_options.strategy_options or [], _options, unsupported=["max_fevals", "time_limit", "x0", "searchspace_construction_options"])

    # call the runner
    return runner.run(searchspace.sorted_list(), tuning_options)


tune.__doc__ = common.get_strategy_docstring("Brute Force", _options)
