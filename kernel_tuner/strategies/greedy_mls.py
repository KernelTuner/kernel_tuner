"""A greedy multi-start local search algorithm for parameter search."""
from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.hillclimbers import base_hillclimb

_options = dict(neighbor=("Method for selecting neighboring nodes, choose from Hamming or adjacent", "Hamming"),
                       restart=("controls greedyness, i.e. whether to restart from a position as soon as an improvement is found", True),
                       order=("set a user-specified order to search among dimensions while hillclimbing", None),
                       randomize=("use a random order to search among dimensions while hillclimbing", True))

def tune(searchspace: Searchspace, runner, tuning_options):

    # retrieve options with defaults
    options = tuning_options.strategy_options
    neighbor, restart, order, randomize = common.get_options(options, _options)

    max_fevals = options.get("max_fevals", 100)

    cost_func = common.CostFunc(searchspace, tuning_options, runner)

    # limit max_fevals to max size of the parameter space
    max_fevals = min(searchspace.size, max_fevals)

    fevals = 0

    #while searching
    while fevals < max_fevals:
        candidate = searchspace.get_random_sample(1)[0]

        try:
            base_hillclimb(candidate, neighbor, max_fevals, searchspace, tuning_options, cost_func, restart=restart, randomize=randomize, order=order)
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)
            return cost_func.results

        fevals = len(tuning_options.unique_results)

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("Greedy Multi-start Local Search (MLS)", _options)
