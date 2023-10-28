import os

import numpy as np
import pytest

import kernel_tuner
from kernel_tuner import util
from kernel_tuner.interface import strategy_map

cache_filename = os.path.dirname(os.path.realpath(__file__)) + "/../test_cache_file.json"

@pytest.fixture
def vector_add():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 100
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    tune_params = dict()
    tune_params["block_size_x"] = [128 + 64 * i for i in range(15)]

    return ["vector_add", kernel_string, size, args, tune_params]


@pytest.mark.parametrize('strategy', strategy_map)
def test_strategies(vector_add, strategy):

    options = dict(popsize=5, neighbor='adjacent')

    print(f"testing {strategy}")

    if hasattr(kernel_tuner.interface.strategy_map[strategy], "_options"):
        filter_options = {opt:val for opt, val in options.items() if opt in kernel_tuner.interface.strategy_map[strategy]._options}
    else:
        filter_options = options
    filter_options["max_fevals"] = 10

    results, _ = kernel_tuner.tune_kernel(*vector_add, strategy=strategy, strategy_options=filter_options,
                                         verbose=False, cache=cache_filename, simulation_mode=True)

    assert len(results) > 0

    # check if the number of valid unique configurations is less than or equal to max_fevals
    if not strategy == "brute_force":
        tune_params = vector_add[-1]
        unique_results = {}
        for result in results:
            x_int = ",".join([str(v) for k, v in result.items() if k in tune_params])
            if not isinstance(result["time"], util.InvalidConfig):
                unique_results[x_int] = result["time"]
        assert len(unique_results) <= filter_options["max_fevals"]

    # check whether the returned dictionaries contain exactly the expected keys and the appropriate type
    expected_items = {
        'block_size_x': int,
        'time': (float, int),
        'times': list,
        'compile_time': (float, int),
        'verification_time': (float, int),
        'benchmark_time': (float, int),
        'strategy_time': (float, int),
        'framework_time': (float, int),
        'timestamp': str
    }
    for res in results:
        assert len(res) == len(expected_items)
        for expected_key, expected_type in expected_items.items():
            assert expected_key in res
            assert isinstance(res[expected_key], expected_type)
