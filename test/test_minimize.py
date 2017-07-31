from collections import OrderedDict

import random
import numpy

from kernel_tuner.interface import Options
import kernel_tuner.strategies.minimize as minimize



def test_get_bounds_x0_eps():

    tuning_options = Options()
    tuning_options["scaling"] = True
    tune_params = OrderedDict()
    tune_params['x'] = [0, 1, 2, 3, 4]

    tuning_options["tune_params"] = tune_params

    bounds, x0, eps = minimize.get_bounds_x0_eps(tuning_options)

    assert bounds == [(0.0, 1.0)]
    assert x0 == [0.5]
    assert eps == 0.2

    tuning_options["scaling"] = False

    bounds, x0, eps = minimize.get_bounds_x0_eps(tuning_options)

    assert bounds == [(0, 4)]
    assert x0 == [2.0]
    assert eps == 1.0




def test_get_bounds():

    tune_params = OrderedDict()
    tune_params['x'] = [0, 1, 2, 3, 4]
    tune_params['y'] = [i for i in range(0, 10000, 100)]
    tune_params['z'] = [-11.2, 55.67, 123.27]

    for k in tune_params.keys():
        random.shuffle(tune_params[k])

    expected = [(0, 4), (0, 9900), (-11.2, 123.27)]
    answer = minimize.get_bounds(tune_params)
    assert answer == expected


def test_snap_to_nearest_config():

    tune_params = OrderedDict()
    tune_params['x'] = [0, 1, 2, 3, 4, 5]
    tune_params['y'] = [0, 1, 2, 3, 4, 5]
    tune_params['z'] = [0, 1, 2, 3, 4, 5]

    x = [-5.7, 3.14, 1e6]
    expected = [0, 3, 5]

    answer = minimize.snap_to_nearest_config(x, tune_params)
    assert answer == expected


def test_unscale():

    params = OrderedDict()
    params['x'] = [2**i for i in range(4, 9)]
    eps = 1.0/len(params['x'])

    assert minimize.unscale_and_snap_to_nearest([0], params, eps)[0] == params['x'][0]
    assert minimize.unscale_and_snap_to_nearest([1], params, eps)[0] == params['x'][-1]

    intervals = numpy.linspace(0, 1, len(params['x'])*10)

    freq = dict()
    for i in intervals:
        v = minimize.unscale_and_snap_to_nearest([i], params, eps)[0]
        if v in freq:
            freq[v] += 1
        else:
            freq[v] = 1
        print(i, v)

    print(freq)

    for k, v in freq.items():
        assert v == freq[params['x'][0]]

    assert len(freq.keys()) == len(params['x'])

