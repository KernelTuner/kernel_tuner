from __future__ import print_function
from collections import OrderedDict
from random import randrange, choice
from math import ceil

try:
    from mock import patch
except ImportError:
    from unittest.mock import patch

from kernel_tuner.interface import Options
from kernel_tuner.searchspace import Searchspace

from constraint import ExactSumConstraint, FunctionConstraint
import numpy as np

max_threads = 1024

# 9 combinations without restrictions
simple_tune_params = OrderedDict()
simple_tune_params['x'] = [1, 2, 3]
simple_tune_params['y'] = [4, 5.5]
simple_tune_params['z'] = ['string_1', 'string_2']
simple_tuning_options = Options(dict(restrictions=[], tune_params=simple_tune_params))
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
    assert simple_searchspace.get_param_indices(last) == (2, 1, 1)


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
        print("Expected a ValueError to be raised")
        assert False
    except ValueError as e:
        assert "number of samples requested is greater than the searchspace size" in str(e)
    except Exception:
        print("Expected a ValueError to be raised")
        assert False


def test_neighbors_hamming():
    """ test whether the neighbors with Hamming distance are as expected """
    test_config = tuple([1, 4, 'string_1'])
    expected_neighbors = [(2, 4, 'string_1'), (3, 4, 'string_1'), (1, 5.5, 'string_1'), (1, 4, 'string_2')]

    # prebuilt
    simple_searchspace_prebuilt = Searchspace(simple_tuning_options, max_threads, build_neighbors_index=True, neighbor_method='Hamming')
    neighbors = simple_searchspace_prebuilt.get_neighbors(test_config)
    assert len(neighbors) == len(expected_neighbors)
    assert test_config not in neighbors
    for neighbor in neighbors:
        assert neighbor in expected_neighbors

    # direct
    neighbors = simple_searchspace.get_neighbors(test_config, 'Hamming')
    assert len(neighbors) == len(expected_neighbors)
    assert test_config not in neighbors
    for neighbor in neighbors:
        assert neighbor in expected_neighbors


def test_neighbors_strictlyadjacent():
    """ test whether the strictly adjacent neighbors are as expected """
    test_config = tuple([1, 4, 'string_1'])
    expected_neighbors = [(2, 5.5, 'string_2'), (1, 5.5, 'string_2'), (2, 5.5, 'string_1'), (1, 5.5, 'string_1'), (2, 4, 'string_2'), (1, 4, 'string_2'),
                          (2, 4, 'string_1')]

    # prebuilt
    simple_searchspace_prebuilt = Searchspace(simple_tuning_options, max_threads, build_neighbors_index=True, neighbor_method='strictly-adjacent')
    neighbors = simple_searchspace_prebuilt.get_neighbors(test_config)
    assert len(neighbors) == len(expected_neighbors)
    assert test_config not in neighbors
    for neighbor in neighbors:
        assert neighbor in expected_neighbors

    # direct
    neighbors = simple_searchspace.get_neighbors(test_config, 'strictly-adjacent')
    assert len(neighbors) == len(expected_neighbors)
    assert test_config not in neighbors
    for neighbor in neighbors:
        assert neighbor in expected_neighbors


def test_neighbors_adjacent():
    """ test whether the adjacent neighbors are as expected """
    pass
