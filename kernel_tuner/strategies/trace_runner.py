"""Special strategy that takes a trace of another strategy and repeats this exactly."""
import json
from pathlib import Path
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import get_options, get_strategy_docstring

_options = dict(
    configurations=("Array of tuples denoting the configurations", []), 
    path_to_cachefile=("Path to a cachefile to repeat the configurations", '')
)


def tune(searchspace: Searchspace, runner, tuning_options):
    args = get_options(tuning_options.strategy_options, _options)
    assert (args[0] != _options[0]) != (args[1] != _options[1]), "Trace runner takes exactly one tuning_options argument"
    if args[0] != _options[0]:
        configurations: list[tuple] = args[0]
    if args[1] != _options[1]:
        path_to_cachefile = Path(args[1])
        assert path_to_cachefile.exists()
        configurations: list[tuple] = list()
        with open(path_to_cachefile, 'rb') as fp:
            cache = dict(dict(json.load(fp))['cache'])
            for key in cache.keys():
                config = list()
                for param in key.split(','):
                    try:
                        param = float(param) if '.' in param else int(param)
                    except ValueError:
                        param = str(param)
                    config.append(param)
                configurations.append(tuple(config))
    assert len(configurations) > 0

    # check that the configurations are in the searchspace
    for configuration in configurations:
        assert searchspace.is_param_config_valid(configuration)

    # call the runner
    return runner.run(configurations, tuning_options)


tune.__doc__ = get_strategy_docstring("Trace Runner", _options)
