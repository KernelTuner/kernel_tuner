import os

import numpy as np
import pytest

import kernel_tuner
from kernel_tuner.util import InvalidConfig
from kernel_tuner.interface import strategy_map

from ..context import skip_if_no_bayesopt_botorch, skip_if_no_bayesopt_gpytorch

cache_filename = os.path.dirname(os.path.realpath(__file__)) + "/test_cache_file.json"

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
    tune_params["test_string"] = ["alg_1", "alg_2"]
    tune_params["test_single"] = [15]
    tune_params["test_bool"] = [True, False]
    tune_params["test_mixed"] = ["test", 1, True, 2.45]

    return ["vector_add", kernel_string, size, args, tune_params]

# skip some strategies if their dependencies are not installed
strategies = []
for s in strategy_map.keys():
    if 'gpytorch' in s.lower() or 'botorch_alt' in s.lower() or 'bayes_opt_old' in s.lower():
        continue
    if 'gpytorch' in s.lower():
        strategies.append(pytest.param(s, marks=skip_if_no_bayesopt_gpytorch))
    elif 'botorch' in s.lower():
        strategies.append(pytest.param(s, marks=skip_if_no_bayesopt_botorch))
    else:
        strategies.append(s)
@pytest.mark.parametrize('strategy', strategies)
def test_strategies(vector_add, strategy):

    options = dict(popsize=5, neighbor='adjacent')

    print(f"testing {strategy}")

    if hasattr(kernel_tuner.interface.strategy_map[strategy], "_options"):
        filter_options = {opt:val for opt, val in options.items() if opt in kernel_tuner.interface.strategy_map[strategy]._options}
    else:
        filter_options = options
    filter_options["max_fevals"] = 10

    restrictions = ["test_string == 'alg_2'", "test_bool == True", "test_mixed == 2.45"]

    results, _ = kernel_tuner.tune_kernel(*vector_add, restrictions=restrictions, strategy=strategy, strategy_options=filter_options,
                                         verbose=False, cache=cache_filename, simulation_mode=True)

    assert len(results) > 0

    # check if the number of valid unique configurations is less than or equal to max_fevals
    if not strategy == "brute_force":
        tune_params = vector_add[-1]
        unique_results = {}
        for result in results:
            x_int = ",".join([str(v) for k, v in result.items() if k in tune_params])
            if not isinstance(result["time"], InvalidConfig):
                unique_results[x_int] = result["time"]
        assert len(unique_results) <= filter_options["max_fevals"]

    # check whether the returned dictionaries contain exactly the expected keys and the appropriate type
    expected_items = {
        'block_size_x': int,
        'test_string': str,
        'test_single': int,
        'test_bool': bool,
        'test_mixed': float,
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
