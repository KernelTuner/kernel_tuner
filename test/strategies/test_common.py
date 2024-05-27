import sys
from time import perf_counter

from kernel_tuner.interface import Options
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc

try:
    from mock import Mock
except ImportError:
    from unittest.mock import Mock


def fake_runner():
    fake_result = {
        'time': 5
    }
    runner = Mock()
    runner.last_strategy_start_time = perf_counter()
    runner.run.return_value = [fake_result]
    return runner


tune_params = dict([("x", [1, 2, 3]), ("y", [4, 5, 6])])


def test_cost_func():
    x = [1, 4]
    tuning_options = Options(scaling=False, snap=False, tune_params=tune_params,
                             restrictions=None, strategy_options={}, cache={}, unique_results={},
                             objective="time", objective_higher_is_better=False, metrics=None)
    runner = fake_runner()

    time = CostFunc(Searchspace(tune_params, None, 1024), tuning_options, runner)(x)
    assert time == 5

    # check if restrictions are properly handled
    def restrictions(_):
        return False
    tuning_options = Options(scaling=False, snap=False, tune_params=tune_params,
                             restrictions=restrictions, strategy_options={},
                             verbose=True, cache={}, unique_results={},
                             objective="time", objective_higher_is_better=False, metrics=None)
    time = CostFunc(Searchspace(tune_params, restrictions, 1024), tuning_options, runner)(x)
    assert time == sys.float_info.max


def test_setup_method_arguments():
    # check if returns a dict, the specific options depend on scipy
    assert isinstance(common.setup_method_arguments("bla", 5), dict)


def test_setup_method_options():
    tuning_options = Options(eps=1e-5, tune_params=tune_params, strategy_options={}, verbose=True)

    method_options = common.setup_method_options("L-BFGS-B", tuning_options)
    assert isinstance(method_options, dict)
    assert method_options["eps"] == 1e-5
    assert method_options["maxfun"] == 100
    assert method_options["disp"] is True
