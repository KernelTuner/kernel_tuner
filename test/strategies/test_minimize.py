from __future__ import print_function

from collections import OrderedDict
import numpy as np

import kernel_tuner
from kernel_tuner.strategies import minimize
from kernel_tuner.interface import Options


try:
    from mock import patch, Mock
except ImportError:
    from unittest.mock import patch, Mock


def fake_runner():
    fake_result = {'time': 5}
    runner = Mock()
    runner.run.return_value = [[fake_result], None]
    return runner

tune_params = OrderedDict(x=[1, 2, 3], y=[4, 5, 6])




def test__cost_func():

    x = [1, 4]
    kernel_options = None
    tuning_options = Options(scaling=False, tune_params=tune_params, restrictions=None)
    runner = fake_runner()
    results = []
    cache = {}

    time = minimize._cost_func(x, kernel_options, tuning_options, runner, results, cache)
    assert time == 5

    time = minimize._cost_func(x, kernel_options, tuning_options, runner, results, cache)
    assert time == 5
    #check if 1st run is properly cached and runner is only called once
    assert runner.run.call_count == 1

    #check if restrictions are properly handled
    restrictions = ["False"]
    cache = {}
    tuning_options = Options(scaling=False,
                             tune_params=tune_params,
                             restrictions=restrictions,
                             verbose=True)
    time = minimize._cost_func(x, kernel_options, tuning_options, runner, results, cache)
    assert time == 1e20


def test_setup_method_arguments():
    #check if returns a dict, the specific options depend on scipy
    assert isinstance(minimize.setup_method_arguments("bla", 5), dict)


def test_setup_method_options():
    tuning_options = Options(eps=1e-5,tune_params=tune_params)

    method_options = minimize.setup_method_options("L-BFGS-B", tuning_options)
    assert isinstance(method_options, dict)
    assert method_options["eps"] == 1e-5
    assert method_options["maxfun"] == 9

