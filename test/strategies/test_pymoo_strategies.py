import os

import numpy as np
import pytest
from pathlib import Path

import kernel_tuner
from kernel_tuner import util

from ..context import skip_if_no_pymoo

cache_filename =  Path(__file__).parent / "test_cache_time_energy.json"

strategies = ["nsga2", "nsga3"]

@skip_if_no_pymoo
@pytest.mark.parametrize('strategy', strategies)
def test_strategies(strategy):

    options = dict(strategy=strategy,
                strategy_options = dict(popsize=5, max_fevals=15),
                objective = ["time", "energy"],
                objective_higher_is_better = [False, False],
                verbose=True,
              )

    print(f"testing {strategy}")
    assert cache_filename.exists()
    results, env = kernel_tuner.tune_cache(cache_filename, **options)

    # assert has results
    assert len(results) > 0

    # assert pareto front is stored in env["best_config"]
    pareto_front = util.get_pareto_results(results, options["objective"], options["objective_higher_is_better"])
    assert len(pareto_front) == len(env["best_config"])
