from __future__ import print_function

from math import ceil
from random import randrange

try:
    from mock import patch
except ImportError:
    from unittest.mock import patch

import numpy as np
from constraint import ExactSumConstraint

from kernel_tuner.interface import Options
from kernel_tuner.searchspace import Searchspace

max_threads = 1024
value_error_expectation_message = "Expected a ValueError to be raised"

# 16 combinations, of 6 which pass the restrictions
simple_tune_params = dict()
simple_tune_params["x"] = [1, 1.5, 2, 3]
simple_tune_params["y"] = [4, 5.5]
simple_tune_params["z"] = ["string_1", "string_2"]
restrict = ["y % x == 1"]
simple_tuning_options = Options(dict(restrictions=restrict, tune_params=simple_tune_params))
simple_searchspace = Searchspace(simple_tune_params, restrict, max_threads)
simple_searchspace_bruteforce = Searchspace(simple_tune_params, restrict, max_threads, framework="bruteforce")

# 3.1 million combinations, of which 10600 pass the restrictions
num_layers = 42
tune_params = dict()
tune_params["gpu1"] = list(range(num_layers))
tune_params["gpu2"] = list(range(num_layers))
tune_params["gpu3"] = list(range(num_layers))
tune_params["gpu4"] = list(range(num_layers))

# each GPU must have at least one layer and the sum of all layers must not exceed the total number of layers

def _min_func(gpu1, gpu2, gpu3, gpu4):
    return min([gpu1, gpu2, gpu3, gpu4]) >= 1


# test two different types of restrictions: a constraint and a callable
assert callable(_min_func)
restrict = [ExactSumConstraint(num_layers), _min_func]

# create the searchspace object
searchspace = Searchspace(tune_params, restrict, max_threads)
searchspace_bruteforce = Searchspace(tune_params, restrict, max_threads, framework="bruteforce")

# 74088 combinations intended to test whether sorting works
sort_tune_params = dict()
sort_tune_params["gpu1"] = list(range(num_layers))
sort_tune_params["gpu2"] = list(range(num_layers))
sort_tune_params["gpu3"] = list(range(num_layers))
searchspace_sort = Searchspace(sort_tune_params, [], max_threads)


def compare_two_searchspace_objects(searchspace_1: Searchspace, searchspace_2: Searchspace):
    """Helper test function to assert that two searchspace objects are identical in outcome."""
    assert searchspace_1.size == searchspace_2.size
    for dict_config in searchspace_1.get_list_dict().keys():
        assert searchspace_2.is_param_config_valid(dict_config)


def test_size():
    """Test that the searchspace after applying restrictions is the expected size."""
    assert simple_searchspace.size == 6
    assert searchspace.size == 10660


def test_internal_representation():
    """Test that the list and dict representations match in size, type and elements."""
    assert searchspace.size == len(searchspace.list)
    assert searchspace.size == len(searchspace.get_list_dict().keys())
    assert isinstance(searchspace.list[0], tuple)

    for index, dict_config in enumerate(searchspace.get_list_dict().keys()):
        assert dict_config == searchspace.list[index]

def test_check_restrictions():
    """Test whether the outcome of restrictions is as expected when using check_restrictions."""
    from kernel_tuner.util import check_restrictions

    param_config_false = {'x': 1, 'y': 4, 'z': "string_1" }
    param_config_true = {'x': 3, 'y': 4, 'z': "string_1" }

    assert check_restrictions(simple_searchspace.restrictions, param_config_false, verbose=False) is False
    assert check_restrictions(simple_searchspace.restrictions, param_config_true, verbose=False) is True


def test_against_bruteforce():
    """Tests the default Searchspace framework against bruteforcing the searchspace."""
    compare_two_searchspace_objects(simple_searchspace, simple_searchspace_bruteforce)
    compare_two_searchspace_objects(searchspace, searchspace_bruteforce)

def test_sort():
    """Test that the sort searchspace option works as expected."""
    simple_searchspace_sort = Searchspace(
        simple_tuning_options.tune_params,
        simple_tuning_options.restrictions,
        max_threads
    )

    expected = [
        (1.5, 4, "string_1"),
        (1.5, 4, "string_2"),
        (1.5, 5.5, "string_1"),
        (1.5, 5.5, "string_2"),
        (3, 4, "string_1"),
        (3, 4, "string_2"),
    ]

    # Check if lists match without considering order
    assert set(simple_searchspace_sort.list) == set(expected)

    # Check if lists match, also considering order
    assert simple_searchspace_sort.sorted_list() == expected

    sorted_list = searchspace_sort.sorted_list(sort_last_param_first=False)
    num_params = len(sorted_list[0])
    for param_config_index, (param_config_first, param_config_second) in enumerate(zip(sorted_list, sorted_list[1:])):
        if (param_config_index + 1) % num_layers == 0:
            continue
        for param_index in range(num_params):
            assert param_config_first[param_index] <= param_config_second[param_index]


def test_sort_reversed():
    """Test that the sort searchspace option with the sort_last_param_first option enabled works as expected."""
    simple_searchspace_sort_reversed = Searchspace(
        simple_tuning_options.tune_params,
        simple_tuning_options.restrictions,
        max_threads
    )

    expected = [
        (1.5, 4, "string_1"),
        (3, 4, "string_1"),
        (1.5, 5.5, "string_1"),
        (1.5, 4, "string_2"),
        (3, 4, "string_2"),
        (1.5, 5.5, "string_2"),
    ]

    # Check if lists match without considering order
    assert set(simple_searchspace_sort_reversed.list) == set(expected)

    # Check if lists match, also considering order
    assert simple_searchspace_sort_reversed.sorted_list(sort_last_param_first=True) == expected

    sorted_list = searchspace_sort.sorted_list(sort_last_param_first=True)
    num_params = len(sorted_list[0])
    for param_config_index, (param_config_first, param_config_second) in enumerate(zip(sorted_list, sorted_list[1:])):
        if (param_config_index + 1) % num_layers == 0:
            continue
        for param_index in range(num_params):
            assert param_config_first[param_index] <= param_config_second[param_index]


def test_index_lookup():
    """Test that index lookups are consistent for ~1% of the searchspace."""
    size = searchspace.size
    for _ in range(ceil(size / 100)):
        random_index = randrange(0, size)
        random_param_config = tuple(searchspace.list[random_index])
        index = searchspace.get_param_config_index(random_param_config)
        assert index == random_index


def test_param_index_lookup():
    """Test the parameter index lookup for a parameter config is as expected."""
    first = tuple([1, 4, "string_1"])
    last = tuple([3, 5.5, "string_2"])
    assert simple_searchspace.get_param_indices(first) == (0, 0, 0)
    assert simple_searchspace.get_param_indices(last) == (3, 1, 1)


def test_random_sample():
    """Test whether the random sample indices exists and are unique, and if it throws an error for too many samples."""
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
        assert "number of samples requested" in str(e) and "is greater than the searchspace size" in str(e), f"Expected string not in error {e}"
    except Exception:
        print(value_error_expectation_message)
        assert False


def __test_neighbors_prebuilt(param_config: tuple, expected_neighbors: list, neighbor_method: str):
    simple_searchspace_prebuilt = Searchspace(
        simple_tuning_options.tune_params,
        simple_tuning_options.restrictions,
        max_threads,
        build_neighbors_index=True,
        neighbor_method=neighbor_method,
    )
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
    """Test whether the neighbors with Hamming distance are as expected."""
    test_config = tuple([1, 4, "string_1"])
    expected_neighbors = [
        (1.5, 4, 'string_1'),
        (3, 4, 'string_1'),
    ]

    __test_neighbors(test_config, expected_neighbors, "Hamming")


def test_neighbors_strictlyadjacent():
    """Test whether the strictly adjacent neighbors are as expected."""
    test_config = tuple([1, 4, "string_1"])
    expected_neighbors = [
        (1.5, 4, 'string_1'),
        (1.5, 4, 'string_2'),
        (1.5, 5.5, 'string_1'),
        (1.5, 5.5, 'string_2'),
    ]

    __test_neighbors(test_config, expected_neighbors, "strictly-adjacent")


def test_neighbors_adjacent():
    """Test whether the adjacent neighbors are as expected."""
    test_config = tuple([1, 4, "string_1"])
    expected_neighbors = [
        (1.5, 4, 'string_1'),
        (1.5, 4, 'string_2'),
        (1.5, 5.5, 'string_1'),
        (1.5, 5.5, 'string_2'),
    ]

    __test_neighbors(test_config, expected_neighbors, "adjacent")


def test_neighbors_fictious():
    """Test whether the neighbors are as expected for a fictious parameter configuration (i.e. not existing in the search space due to restrictions)."""
    test_config = tuple([1.5, 4, "string_1"])
    expected_neighbors_hamming = [
        (1.5, 4, 'string_2'),
        (1.5, 5.5, 'string_1'),
        (3, 4, 'string_1'),
    ]
    expected_neighbors_strictlyadjacent = [
        (1.5, 5.5, 'string_2'),
        (1.5, 5.5, 'string_1'),
        (1.5, 4, 'string_2')
    ]

    expected_neighbors_adjacent = [
        (1.5, 5.5, 'string_2'),
        (1.5, 5.5, 'string_1'),
        (1.5, 4, 'string_2'),
        (3, 4, 'string_1'),
        (3, 4, 'string_2'),
    ]

    __test_neighbors_direct(test_config, expected_neighbors_hamming, "Hamming")
    __test_neighbors_direct(test_config, expected_neighbors_strictlyadjacent, "strictly-adjacent")
    __test_neighbors_direct(test_config, expected_neighbors_adjacent, "adjacent")


def test_neighbors_cached():
    """Test whether retrieving a set of neighbors twice returns the cached version."""
    simple_searchspace_duplicate = Searchspace(
        simple_tuning_options.tune_params,
        simple_tuning_options.restrictions,
        max_threads,
        neighbor_method="Hamming"
    )

    test_configs = simple_searchspace_duplicate.get_random_sample(5)
    for test_config in test_configs:
        assert not simple_searchspace_duplicate.are_neighbors_indices_cached(test_config)
        neighbors = simple_searchspace_duplicate.get_neighbors(test_config)
        assert simple_searchspace_duplicate.are_neighbors_indices_cached(test_config)
        neighbors_2 = simple_searchspace_duplicate.get_neighbors(test_config)
        assert neighbors == neighbors_2


def test_param_neighbors():
    """Test whether for a given parameter configuration and index the correct neighboring parameters are returned."""
    test_config = tuple([1.5, 4, "string_1"])
    expected_neighbors = [[3], [5.5], ["string_2"]]

    for index in range(3):
        neighbor_params = simple_searchspace.get_param_neighbors(test_config, index, "adjacent", randomize=False)
        assert len(neighbor_params) == len(expected_neighbors[index])
        for param_index, param in enumerate(neighbor_params):
            assert param == expected_neighbors[index][param_index]


@patch("kernel_tuner.searchspace.choice", lambda x: x[0])
def test_order_param_configs():
    """Test whether the ordering of parameter configurations according to parameter index happens as expected."""
    test_order = [1, 2, 0]
    test_config = tuple([1, 4, "string_1"])
    expected_order = [
        (1.5, 5.5, 'string_2'),
        (1.5, 4, 'string_2'),
        (1.5, 4, 'string_1'),
        (1.5, 5.5, 'string_1')
    ]
    neighbors = simple_searchspace.get_neighbors_no_cache(test_config, "adjacent")

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
        assert expected_param_config in ordered_neighbors
        assert expected_param_config == ordered_neighbors[index]

    # test randomize in params
    ordered_neighbors = simple_searchspace.order_param_configs(neighbors, test_order, randomize_in_params=True)
    for expected_param_config in expected_order:
        assert expected_param_config in ordered_neighbors
    assert len(ordered_neighbors) == len(expected_order)


def test_small_searchspace():
    """Test a small real-world searchspace and the usage of the `max_threads` parameter."""
    max_threads = 1024
    tune_params = dict()
    tune_params["block_size_x"] = [1, 2, 4, 8, 16] + [32*i for i in range(1,33)]
    tune_params["block_size_y"] = [2**i for i in range(6)]
    tune_params["tile_size_x"] = [i for i in range(1,11)]
    restrictions = [
        "block_size_x*block_size_y >= 32",
        f"block_size_x*block_size_y <= {max_threads}",
    ]
    searchspace = Searchspace(tune_params, restrictions, max_threads)
    searchspace_bruteforce = Searchspace(tune_params, restrictions, max_threads, framework="bruteforce")
    compare_two_searchspace_objects(searchspace, searchspace_bruteforce)

def test_full_searchspace(compare_against_bruteforce=False):
    """Tests a full real-world searchspace (expdist). If `compare_against_bruteforce`, the searcspace will be bruteforced to compare against, this can take a long time!."""
    # device characteristics
    dev = {
        'device_name': 'NVIDIA A40',
        'max_threads': 1024,
        'max_shared_memory_per_block': 49152,
        'max_shared_memory': 102400
    }

    # tunable parameters and restrictions
    tune_params = dict()
    tune_params["block_size_x"] = [1, 2, 4, 8, 16] + [32*i for i in range(1,33)]
    tune_params["block_size_y"] = [2**i for i in range(6)]
    tune_params["tile_size_x"] = [i for i in range(1,11)]
    tune_params["tile_size_y"] = [i for i in range(1,11)]
    tune_params["temporal_tiling_factor"] = [i for i in range(1,11)]
    max_tfactor = max(tune_params["temporal_tiling_factor"])
    tune_params["max_tfactor"] = [max_tfactor]
    tune_params["loop_unroll_factor_t"] = [i for i in range(1,max_tfactor+1)]
    tune_params["sh_power"] = [0,1]
    tune_params["blocks_per_sm"] = [0,1,2,3,4]

    restrictions = [
            "block_size_x*block_size_y >= 32",
            "temporal_tiling_factor % loop_unroll_factor_t == 0",
            f"block_size_x*block_size_y <= {dev['max_threads']}",
            f"(block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4 <= {dev['max_shared_memory_per_block']}",
            f"blocks_per_sm == 0 or (((block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4) * blocks_per_sm <= {dev['max_shared_memory']})"
        ]

    # build the searchspace
    searchspace = Searchspace(tune_params, restrictions, max_threads=dev['max_threads'])

    if compare_against_bruteforce:
        searchspace_bruteforce = Searchspace(tune_params, restrictions, max_threads=dev['max_threads'], framework='bruteforce')
        compare_two_searchspace_objects(searchspace, searchspace_bruteforce)
    else:
        assert searchspace.size == len(searchspace.list) == 349853
