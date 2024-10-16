from kernel_tuner.hyper import tune_hyper_params

from .context import skip_if_no_methodology
from .test_runners import cache_filename, env  # noqa: F401


@skip_if_no_methodology
def test_hyper(env):

    hyper_params = dict()
    hyper_params["popsize"] = [5]
    hyper_params["maxiter"] = [5, 10]
    hyper_params["method"] = ["uniform"]
    hyper_params["mutation_chance"] = [10]

    target_strategy = "genetic_algorithm"

    result = tune_hyper_params(target_strategy, hyper_params, iterations=1, verbose=True, cache=cache_filename)
    raise ValueError(result)
    assert len(result) > 0
