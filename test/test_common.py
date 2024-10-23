import random

import numpy as np

import kernel_tuner.strategies.common as common
from kernel_tuner.interface import Options
from kernel_tuner.searchspace import Searchspace


def test_get_bounds_x0_eps():
    tune_params = dict()
    tune_params['x'] = [0, 1, 2, 3, 4]
    searchspace = Searchspace(tune_params, [], 1024)

    tuning_options = Options()
    tuning_options["strategy_options"] = {}

    bounds, x0, eps = common.CostFunc(searchspace, tuning_options, None, scaling=True).get_bounds_x0_eps()

    assert bounds == [(0.0, 1.0)]
    assert x0 >= 0.0 and x0 <= 1.0
    assert eps == 0.2

    bounds, x0, eps = common.CostFunc(searchspace, tuning_options, None, scaling=False).get_bounds_x0_eps()

    assert bounds == [(0, 4)]
    assert eps == 1.0


def test_get_bounds():

    tune_params = dict()
    tune_params['x'] = [0, 1, 2, 3, 4]
    tune_params['y'] = [i for i in range(0, 10000, 100)]
    tune_params['z'] = [-11.2, 55.67, 123.27]

    for k in tune_params.keys():
        random.shuffle(tune_params[k])

    expected = [(0, 4), (0, 9900), (-11.2, 123.27)]
    searchspace = Searchspace(tune_params, None, 1024)
    cost_func = common.CostFunc(searchspace, None, None)
    answer = cost_func.get_bounds()
    assert answer == expected


def test_snap_to_nearest_config():

    tune_params = dict()
    tune_params['x'] = [0, 1, 2, 3, 4, 5]
    tune_params['y'] = [0, 1, 2, 3, 4, 5]
    tune_params['z'] = [0, 1, 2, 3, 4, 5]
    tune_params['w'] = ['a', 'b', 'c']

    x = [-5.7, 3.14, 1e6, 'b']
    expected = [0, 3, 5, 'b']

    answer = common.snap_to_nearest_config(x, tune_params)
    assert answer == expected


def test_unscale():

    params = dict()
    params['x'] = [2**i for i in range(4, 9)]
    eps = 1.0 / len(params['x'])

    assert common.unscale_and_snap_to_nearest([0], params, eps)[0] == params['x'][0]
    assert common.unscale_and_snap_to_nearest([1], params, eps)[0] == params['x'][-1]

    intervals = np.linspace(0, 1, len(params['x']) * 10)

    freq = dict()
    for i in intervals:
        v = common.unscale_and_snap_to_nearest([i], params, eps)[0]
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
        print(i, v)

    print(freq)

    for v in freq.values():
        assert v == freq[params['x'][0]]

    assert len(freq.keys()) == len(params['x'])
