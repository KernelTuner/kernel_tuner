import numpy as np
import pytest
from kernel_tuner.strategies.diff_evo import (
    values_to_indices,
    indices_to_values,
    mutate_de_1,
    mutate_de_2,
    binomial_crossover,
    exponential_crossover,
    parse_method,
    mutation,
    crossover,
)
from kernel_tuner.strategies.diff_evo import supported_methods
from kernel_tuner import tune_kernel

from .test_strategies import vector_add, cache_filename


def test_values_to_indices():

    tune_params = {}
    tune_params["block_size_x"] = [16, 32, 128, 1024]

    result = values_to_indices([1024], tune_params)
    expected = [3]
    assert result[0] == expected[0]
    assert len(result) == len(expected)

    tune_params["block_size_y"] = [16, 32, 128, 1024]

    result = values_to_indices([32, 128], tune_params)
    expected = [1, 2]
    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert len(result) == len(expected)


def test_indices_to_values():

    tune_params = {}
    tune_params["block_size_x"] = [16, 32, 128, 1024]

    expected = [1024]
    result = indices_to_values([3], tune_params)
    assert result[0] == expected[0]
    assert len(result) == len(expected)

    tune_params["block_size_y"] = [16, 32, 128, 1024]
    expected = [1024, 32]
    result = indices_to_values([3, 1], tune_params)
    assert result[0] == expected[0]
    assert result[1] == expected[1]
    assert len(result) == len(expected)


def test_mutate_de_1():

    tune_params = {}
    tune_params["block_size_x"] = [16, 32, 128, 256, 512, 1024]
    tune_params["block_size_y"] = [1, 2, 8]
    tune_params["block_size_z"] = [1, 2, 4, 8]

    a_idx = np.array([0, 1, 2])
    b_idx = np.array([4, 1, 0])
    c_idx = np.array([5, 0, 1])
    randos_idx = [a_idx, b_idx, c_idx]

    F = 0.8
    params_list = list(tune_params)
    min_idx = np.zeros(len(tune_params))
    max_idx = [len(v) - 1 for v in tune_params.values()]

    mutant = mutate_de_1(a_idx, randos_idx, F, min_idx, max_idx, False)

    assert len(mutant) == len(a_idx)

    for dim, idx in enumerate(mutant):
        assert isinstance(idx, np.integer)
        assert min_idx[dim] <= idx <= max_idx[dim]

    mutant = mutate_de_1(a_idx, randos_idx[:-1], F, min_idx, max_idx, True)

    assert len(mutant) == len(a_idx)

    for dim, idx in enumerate(mutant):
        assert isinstance(idx, np.integer)
        assert min_idx[dim] <= idx <= max_idx[dim]


def test_mutate_de_2():

    tune_params = {}
    tune_params["block_size_x"] = [16, 32, 128, 256, 512, 1024]
    tune_params["block_size_y"] = [1, 2, 8]
    tune_params["block_size_z"] = [1, 2, 4, 8]

    a_idx = np.array([0, 1, 2])
    b_idx = np.array([4, 1, 0])
    c_idx = np.array([5, 0, 1])
    d_idx = np.array([3, 2, 3])
    e_idx = np.array([1, 0, 3])
    randos_idx = [a_idx, b_idx, c_idx, d_idx, e_idx]

    F = 0.8
    params_list = list(tune_params)
    min_idx = np.zeros(len(tune_params))
    max_idx = [len(v) - 1 for v in tune_params.values()]

    mutant = mutate_de_2(a_idx, randos_idx, F, min_idx, max_idx, False)

    assert len(mutant) == len(a_idx)

    for dim, idx in enumerate(mutant):
        assert isinstance(idx, np.integer)
        assert min_idx[dim] <= idx <= max_idx[dim]

    mutant = mutate_de_2(a_idx, randos_idx[:-1], F, min_idx, max_idx, True)

    assert len(mutant) == len(a_idx)

    for dim, idx in enumerate(mutant):
        assert isinstance(idx, np.integer)
        assert min_idx[dim] <= idx <= max_idx[dim]


def test_binomial_crossover():

    donor_vector = np.array([1, 2, 3, 4, 5])
    target = np.array([6, 7, 8, 9, 10])
    CR = 0.8

    result = binomial_crossover(donor_vector, target, CR)
    assert len(result) == len(donor_vector)

    for dim, val in enumerate(result):
        assert (val == donor_vector[dim]) or (val == target[dim])


def test_exponential_crossover():

    donor_vector = np.array([1, 2, 3, 4, 5])
    target = np.array([6, 7, 8, 9, 10])
    CR = 0.8

    result = exponential_crossover(donor_vector, target, CR)
    assert len(result) == len(donor_vector)

    for dim, val in enumerate(result):
        assert (val == donor_vector[dim]) or (val == target[dim])


def test_parse_method():

    # check unsupported methods raise ValueError
    for method in ["randtobest4bin", "bogus3log"]:
        print(f"{method=}")
        with pytest.raises(ValueError):
            parse_method(method)

    # check if parses correctly
    def check_result(result, expected):
        assert len(result) == len(expected)
        for i, res in enumerate(result):
            assert res == expected[i]

    check_result(parse_method("rand1bin"), [False, 1, mutation["1"], crossover["bin"]])
    check_result(parse_method("best1exp"), [True, 1, mutation["1"], crossover["exp"]])
    check_result(parse_method("randtobest1exp"), [False, 1, mutation["randtobest"], crossover["exp"]])
    check_result(parse_method("currenttobest1bin"), [False, 1, mutation["currenttobest"], crossover["bin"]])


@pytest.mark.parametrize("method", supported_methods)
def test_diff_evo(vector_add, method):
    restrictions = [
        "test_string == 'alg_2'", 
        "test_bool == True", 
        "test_mixed == 2.45"
    ]
    result, _ = tune_kernel(
        *vector_add,
        restrictions=restrictions,
        strategy="diff_evo",
        strategy_options=dict(popsize=5, method=method),
        verbose=True,
        cache=cache_filename,
        simulation_mode=True,
    )
    assert len(result) > 0
