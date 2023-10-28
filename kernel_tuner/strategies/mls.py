"""The strategy that uses multi-start local search."""
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.greedy_mls import tune as mls_tune

_options = dict(neighbor=("Method for selecting neighboring nodes, choose from Hamming or adjacent", "Hamming"),
                       restart=("controls greedyness, i.e. whether to restart from a position as soon as an improvement is found", False),
                       order=("set a user-specified order to search among dimensions while hillclimbing", None),
                       randomize=("use a random order to search among dimensions while hillclimbing", True))

def tune(searchspace: Searchspace, runner, tuning_options):

    # Default MLS uses 'best improvement' hillclimbing, so greedy hillclimbing is disabled with restart defaulting to False
    _, restart, _, _ = common.get_options(tuning_options.strategy_options, _options)

    # Delegate to greedy_mls.tune() but make sure restart uses our default, if not overwritten by the user
    tuning_options.strategy_options["restart"] = restart
    return mls_tune(searchspace, runner, tuning_options)


tune.__doc__ = common.get_strategy_docstring("Multi-start Local Search (MLS)", _options)
