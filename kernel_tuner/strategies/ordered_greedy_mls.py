"""A greedy multi-start local search algorithm for parameter search that traverses variables in order."""
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.greedy_mls import tune as mls_tune

_options = dict(neighbor=("Method for selecting neighboring nodes, choose from Hamming or adjacent", "Hamming"),
                       restart=("controls greedyness, i.e. whether to restart from a position as soon as an improvement is found", True),
                       order=("set a user-specified order to search among dimensions while hillclimbing", None),
                       randomize=("use a random order to search among dimensions while hillclimbing", False))

def tune(searchspace: Searchspace, runner, tuning_options):

    _, restart, _, randomize = common.get_options(tuning_options.strategy_options, _options)

    # Delegate to Greedy MLS, but make sure our defaults are used if not overwritten by the user
    tuning_options.strategy_options["restart"] = restart
    tuning_options.strategy_options["randomize"] = randomize
    return mls_tune(searchspace, runner, tuning_options)


tune.__doc__ = common.get_strategy_docstring("Ordered Greedy Multi-start Local Search (MLS)", _options)
