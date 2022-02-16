from collections import OrderedDict
import os

import pytest
import numpy as np

import kernel_tuner
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
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [128 + 64 * i for i in range(15)]

    return ["vector_add", kernel_string, size, args, tune_params]


@pytest.mark.parametrize('strategy', strategy_map)
def test_strategies(vector_add, strategy):

    options = dict(popsize=5, max_fevals=15)

    print(f"testing {strategy}")
    result, _ = kernel_tuner.tune_kernel(*vector_add, strategy=strategy, strategy_options=options,
                                             verbose=False, cache=cache_filename, simulation_mode=True)
    assert len(result) > 0
