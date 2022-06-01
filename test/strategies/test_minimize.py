import sys
from collections import OrderedDict
from time import perf_counter
from kernel_tuner.strategies import minimize
from kernel_tuner.interface import Options

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
    runner.run.return_value = [[fake_result], None]
    return runner


tune_params = OrderedDict([("x", [1, 2, 3]), ("y", [4, 5, 6])])


def test__cost_func():

    x = [1, 4]
    kernel_options = None
    tuning_options = Options(scaling=False, snap=False, tune_params=tune_params,
                             restrictions=None, strategy_options={}, cache={}, unique_results={},
                             objective="time", objective_higher_is_better=False)
    runner = fake_runner()
    results = []

    time = minimize._cost_func(x, kernel_options, tuning_options, runner, results)
    assert time == 5

    tuning_options.cache["1,4"] = OrderedDict([("x", 1), ("y", 4), ("time", 5)])

    time = minimize._cost_func(x, kernel_options, tuning_options, runner, results)

    assert time == 5
    # check if 1st run is properly cached and runner is only called once
    assert runner.run.call_count == 1

    # check if restrictions are properly handled
    restrictions = ["False"]
    tuning_options = Options(scaling=False, snap=False, tune_params=tune_params,
                             restrictions=restrictions, strategy_options={},
                             verbose=True, cache={}, unique_results={},
                             objective="time", objective_higher_is_better=False)
    time = minimize._cost_func(x, kernel_options, tuning_options, runner, results)
    assert time == sys.float_info.max


def test_setup_method_arguments():
    # check if returns a dict, the specific options depend on scipy
    assert isinstance(minimize.setup_method_arguments("bla", 5), dict)


def test_setup_method_options():
    tuning_options = Options(eps=1e-5, tune_params=tune_params, strategy_options={}, verbose=True)

    method_options = minimize.setup_method_options("L-BFGS-B", tuning_options)
    assert isinstance(method_options, dict)
    assert method_options["eps"] == 1e-5
    assert method_options["maxfun"] == 100
    assert method_options["disp"] is True
