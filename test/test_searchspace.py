from __future__ import print_function
from collections import OrderedDict
from random import randrange, choice
from math import ceil
from time import perf_counter

try:
    from mock import patch
except ImportError:
    from unittest.mock import patch

from kernel_tuner.interface import Options
from kernel_tuner.searchspace import Searchspace

from constraint import ExactSumConstraint, FunctionConstraint
import numpy as np

max_threads = 1024
value_error_expectation_message = "Expected a ValueError to be raised"

# 9 combinations without restrictions
simple_tune_params = OrderedDict()
simple_tune_params['x'] = [1, 1.5, 2, 3]
simple_tune_params['y'] = [4, 5.5]
simple_tune_params['z'] = ['string_1', 'string_2']
restrict = [lambda x, y, z: x != 1.5]
simple_tuning_options = Options(dict(restrictions=restrict, tune_params=simple_tune_params))
simple_searchspace = Searchspace(simple_tuning_options, max_threads)

# 3.1 million combinations, of which 10600 pass the restrictions
num_layers = 42
tune_params = OrderedDict()
tune_params["gpu1"] = list(range(num_layers))
tune_params["gpu2"] = list(range(num_layers))
tune_params["gpu3"] = list(range(num_layers))
tune_params["gpu4"] = list(range(num_layers))

# each GPU must have at least one layer and the sum of all layers must not exceed the total number of layers
min_func = lambda gpu1, gpu2, gpu3, gpu4: min([gpu1, gpu2, gpu3, gpu4]) >= 1
# test three different types of restrictions: python-constraint, a function and a string
restrict = [ExactSumConstraint(num_layers), FunctionConstraint(min_func)]

# 74088 combinations intended to test whether sorting works
sort_tune_params = OrderedDict()
sort_tune_params["gpu1"] = list(range(num_layers))
sort_tune_params["gpu2"] = list(range(num_layers))
sort_tune_params["gpu3"] = list(range(num_layers))
sort_tuning_options = Options(dict(restrictions=[], tune_params=sort_tune_params))

# create the searchspace object
tuning_options = Options(dict(restrictions=restrict, tune_params=tune_params))
searchspace = Searchspace(tuning_options, max_threads)


def test_size():
    """ test that the searchspace after applying restrictions is the expected size """
    assert simple_searchspace.size == 12
    assert searchspace.size == 10660


def test_internal_representation():
    """ test that the list and dict representations match in size, type and elements """
    assert searchspace.size == len(searchspace.list)
    assert searchspace.size == len(searchspace.get_list_dict().keys())
    assert isinstance(searchspace.list[0], tuple)

    for index, dict_config in enumerate(searchspace.get_list_dict().keys()):
        assert dict_config == searchspace.list[index]

def test_sort():
    """ test that the sort searchspace option works as expected """
    simple_searchspace_sort = Searchspace(simple_tuning_options, max_threads, sort=True, sort_last_param_first=False)
    assert simple_searchspace_sort.list == [(1, 4, 'string_1'), (1, 4, 'string_2'), (1, 5.5, 'string_1'), (1, 5.5, 'string_2'), (2, 4, 'string_1'), (2, 4, 'string_2'), (2, 5.5, 'string_1'), (2, 5.5, 'string_2'), (3, 4, 'string_1'), (3, 4, 'string_2'), (3, 5.5, 'string_1'), (3, 5.5, 'string_2')]

    searchspace_sort = Searchspace(sort_tuning_options, max_threads, sort=True, sort_last_param_first=False)
    num_params = len(searchspace_sort.list[0])
    for param_config_index, (param_config_first, param_config_second) in enumerate(zip(searchspace_sort.list, searchspace_sort.list[1:])):
        if (param_config_index + 1) % num_layers == 0:
            continue
        for param_index in range(num_params):
            assert param_config_first[param_index] <= param_config_second[param_index]

def test_sort_reversed():
    """ test that the sort searchspace option with the sort_last_param_first option enabled works as expected """
    simple_searchspace_sort_reversed = Searchspace(simple_tuning_options, max_threads, sort=True, sort_last_param_first=True)
    assert simple_searchspace_sort_reversed.list == [(1, 4, 'string_1'), (2, 4, 'string_1'), (3, 4, 'string_1'), (1, 5.5, 'string_1'), (2, 5.5, 'string_1'), (3, 5.5, 'string_1'), (1, 4, 'string_2'), (2, 4, 'string_2'), (3, 4, 'string_2'), (1, 5.5, 'string_2'), (2, 5.5, 'string_2'), (3, 5.5, 'string_2')]

    searchspace_sort = Searchspace(sort_tuning_options, max_threads, sort=True, sort_last_param_first=True)
    num_params = len(searchspace_sort.list[0])
    for param_config_index, (param_config_first, param_config_second) in enumerate(zip(searchspace_sort.list, searchspace_sort.list[1:])):
        if (param_config_index + 1) % num_layers == 0:
            continue
        for param_index in range(num_params):
            assert param_config_first[param_index] <= param_config_second[param_index]

def test_index_lookup():
    """ test that index lookups are consistent for ~1% of the searchspace """
    size = searchspace.size
    for _ in range(ceil(size / 100)):
        random_index = randrange(0, size)
        random_param_config = tuple(searchspace.list[random_index])
        index = searchspace.get_param_config_index(random_param_config)
        assert index == random_index


def test_param_index_lookup():
    """ test the parameter index lookup for a parameter config is as expected """
    first = tuple([1, 4, 'string_1'])
    last = tuple([3, 5.5, 'string_2'])
    assert simple_searchspace.get_param_indices(first) == (0, 0, 0)
    assert simple_searchspace.get_param_indices(last) == (3, 1, 1)


def test_random_sample():
    """ test whether the random sample indices exists and are unique, and if it throws an error for too many samples """
    random_sample_indices = searchspace.get_random_sample_indices(100)
    assert len(random_sample_indices) == 100
    for index in random_sample_indices:
        assert isinstance(searchspace.list[index], tuple)
    assert random_sample_indices.size == np.unique(random_sample_indices).size

    random_samples = searchspace.get_random_sample(100)
    for sample in random_samples:
        assert sample in searchspace.list

    # num_samples equal to the number of configs should return the list
    simple_random_sample_indices = simple_searchspace.get_random_sample_indices(simple_searchspace.size)
    assert simple_random_sample_indices.size == simple_searchspace.size
    assert simple_random_sample_indices.size == np.unique(simple_random_sample_indices).size

    # too many samples should result in a ValueError
    try:
        simple_searchspace.get_random_sample_indices(simple_searchspace.size + 1)
        print(value_error_expectation_message)
        assert False
    except ValueError as e:
        assert "number of samples requested is greater than the searchspace size" in str(e)
    except Exception:
        print(value_error_expectation_message)
        assert False


def __test_neighbors_prebuilt(param_config: tuple, expected_neighbors: list, neighbor_method: str):
    simple_searchspace_prebuilt = Searchspace(simple_tuning_options, max_threads, build_neighbors_index=True, neighbor_method=neighbor_method)
    neighbors = simple_searchspace_prebuilt.get_neighbors_no_cache(param_config)
    assert param_config not in neighbors
    for neighbor in neighbors:
        assert neighbor in expected_neighbors
    assert len(neighbors) == len(expected_neighbors)


def __test_neighbors_direct(param_config: tuple, expected_neighbors: list, neighbor_method: str):
    neighbors = simple_searchspace.get_neighbors_no_cache(param_config, neighbor_method)
    assert param_config not in neighbors
    for neighbor in neighbors:
        assert neighbor in expected_neighbors
    assert len(neighbors) == len(expected_neighbors)


def __test_neighbors(param_config: tuple, expected_neighbors: list, neighbor_method: str):
    __test_neighbors_prebuilt(param_config, expected_neighbors, neighbor_method)
    __test_neighbors_direct(param_config, expected_neighbors, neighbor_method)


def test_neighbors_hamming():
    """ test whether the neighbors with Hamming distance are as expected """
    test_config = tuple([1, 4, 'string_1'])
    expected_neighbors = [(2, 4, 'string_1'), (3, 4, 'string_1'), (1, 5.5, 'string_1'), (1, 4, 'string_2')]
    __test_neighbors(test_config, expected_neighbors, 'Hamming')


def test_neighbors_strictlyadjacent():
    """ test whether the strictly adjacent neighbors are as expected """
    test_config = tuple([1, 4, 'string_1'])
    expected_neighbors = [(1, 5.5, 'string_2'), (1, 5.5, 'string_1'), (1, 4, 'string_2')]

    __test_neighbors(test_config, expected_neighbors, 'strictly-adjacent')


def test_neighbors_adjacent():
    """ test whether the adjacent neighbors are as expected """
    test_config = tuple([1, 4, 'string_1'])
    expected_neighbors = [(2, 5.5, 'string_2'), (1, 5.5, 'string_2'), (2, 5.5, 'string_1'), (1, 5.5, 'string_1'), (2, 4, 'string_2'), (1, 4, 'string_2'),
                          (2, 4, 'string_1')]

    __test_neighbors(test_config, expected_neighbors, 'adjacent')


def test_neighbors_fictious():
    """ test whether the neighbors are as expected for a fictious parameter configuration (i.e. not existing in the search space due to restrictions) """
    test_config = tuple([1.5, 4, 'string_1'])
    expected_neighbors_hamming = [(1, 4, 'string_1'), (2, 4, 'string_1'), (3, 4, 'string_1')]
    expected_neighbors_strictlyadjacent = [(2, 5.5, 'string_2'), (1, 5.5, 'string_2'), (2, 5.5, 'string_1'), (1, 5.5, 'string_1'), (2, 4, 'string_2'),
                                           (1, 4, 'string_2'), (2, 4, 'string_1'), (1, 4, 'string_1')]

    expected_neighbors_adjacent = expected_neighbors_strictlyadjacent

    __test_neighbors_direct(test_config, expected_neighbors_hamming, 'Hamming')
    __test_neighbors_direct(test_config, expected_neighbors_strictlyadjacent, 'strictly-adjacent')
    __test_neighbors_direct(test_config, expected_neighbors_adjacent, 'adjacent')


def test_neighbors_cached():
    """ test whether retrieving a set of neighbors twice returns the cached version """
    simple_searchspace_duplicate = Searchspace(simple_tuning_options, max_threads, neighbor_method='Hamming')
    test_configs = simple_searchspace_duplicate.get_random_sample(10)
    for test_config in test_configs:
        start_time = perf_counter()
        neighbors = simple_searchspace_duplicate.get_neighbors(test_config)
        time_first = perf_counter() - start_time
        start_time = perf_counter()
        neighbors_2 = simple_searchspace_duplicate.get_neighbors(test_config)
        time_second = perf_counter() - start_time
        assert neighbors == neighbors_2
        if abs(time_first - time_second) > 1e-7:
            assert time_second < time_first


def test_param_neighbors():
    """ test whether for a given parameter configuration and index the correct neighboring parameters are returned """
    test_config = tuple([1.5, 4, 'string_1'])
    expected_neighbors = [[1, 2], [5.5], ['string_2']]

    for index in range(3):
        neighbor_params = simple_searchspace.get_param_neighbors(test_config, index, 'adjacent', randomize=False)
        print(neighbor_params)
        assert len(neighbor_params) == len(expected_neighbors[index])
        for param_index, param in enumerate(neighbor_params):
            assert param == expected_neighbors[index][param_index]


@patch('kernel_tuner.searchspace.choice', lambda x: x[0])
def test_order_param_configs():
    """ test whether the ordering of parameter configurations according to parameter index happens as expected """
    test_order = [1, 2, 0]
    test_config = tuple([1, 4, 'string_1'])
    expected_order = [(2, 5.5, 'string_2'), (2, 4, 'string_2'), (1, 4, 'string_2'), (2, 4, 'string_1'), (2, 5.5, 'string_1'), (1, 5.5, 'string_1'),
                      (1, 5.5, 'string_2')]
    neighbors = simple_searchspace.get_neighbors_no_cache(test_config, 'adjacent')

    # test failsafe too few indices
    try:
        simple_searchspace.order_param_configs(neighbors, [1, 2])
        print(value_error_expectation_message)
        assert False
    except ValueError as e:
        assert "must be equal to the number of parameters" in str(e)
    except Exception:
        print(value_error_expectation_message)
        assert False

    # test failsafe too many indices
    try:
        simple_searchspace.order_param_configs(neighbors, [1, 2, 0, 2])
        print(value_error_expectation_message)
        assert False
    except ValueError as e:
        assert "must be equal to the number of parameters" in str(e)
    except Exception:
        print(value_error_expectation_message)
        assert False

    # test failsafe invalid indices
    try:
        simple_searchspace.order_param_configs(neighbors, [1, 3, 0])
        print(value_error_expectation_message)
        assert False
    except ValueError as e:
        assert "order needs to be a list of the parameter indices, but index" in str(e)
    except Exception:
        print(value_error_expectation_message)
        assert False

    # test usecase
    ordered_neighbors = simple_searchspace.order_param_configs(neighbors, test_order, randomize_in_params=False)
    for index, expected_param_config in enumerate(expected_order):
        assert expected_param_config == ordered_neighbors[index]

    # test randomize in params
    ordered_neighbors = simple_searchspace.order_param_configs(neighbors, test_order, randomize_in_params=True)
    for expected_param_config in expected_order:
        assert expected_param_config in ordered_neighbors
    assert len(ordered_neighbors) == len(expected_order)


def test_max_threads():
    max_threads = 1024
    tune_params = dict()
    tune_params["block_size_x"] = [512, 1024]
    tune_params["block_size_y"] = [1]
    tuning_options = Options(dict(tune_params=tune_params, restrictions=None))

    searchspace = Searchspace(tuning_options, max_threads)

    print(searchspace.list)

    assert len(searchspace.list) > 1
