"""Special strategy that takes a trace of another strategy and repeats this exactly."""
import json
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import get_options, get_strategy_docstring

_options = dict(configurations=("Array of tuples denoting the configurations", []), path_to_configurations=("Path to a JSON trace file containing configurations", ''))


def tune(searchspace: Searchspace, runner, tuning_options):
    args = get_options(tuning_options.strategy_options, _options)
    assert (args[0] != _options[0]) != (args[1] != _options[1]), "Trace runner takes exactly one tuning_options argument"
    if args[0] != _options[0]:
        configurations: list[tuple] = args[0]
    if args[1] != _options[1]:
        path_to_configurations = args[1]
        with open(path_to_configurations, 'rb') as fp:
            configurations = [tuple(c) for c in list(json.load(fp))]
    assert len(configurations) > 0

    # check that the configurations are in the searchspace
    for configuration in configurations:
        assert searchspace.is_param_config_valid(configuration)

    # call the runner
    return runner.run(configurations, tuning_options)


tune.__doc__ = get_strategy_docstring("Trace Runner", _options)
