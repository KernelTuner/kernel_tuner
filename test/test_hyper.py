from kernel_tuner.hyper import tune_hyper_params

from .context import skip_if_no_methodology
from .test_runners import env  # noqa: F401


@skip_if_no_methodology
def test_hyper(env):

    hyper_params = dict()
    hyper_params["popsize"] = [5]
    hyper_params["maxiter"] = [5, 10]
    hyper_params["method"] = ["uniform"]
    hyper_params["mutation_chance"] = [10]

    target_strategy = "genetic_algorithm"

    compiler_options = {
        "gpus": ["A100", "MI250X"],
        "override": { 
            "experimental_groups_defaults": { 
                "repeats": 1,
                "samples": 1,
                "minimum_fraction_of_budget_valid": 0.01, 
            },
            "statistics_settings": {
                "cutoff_percentile": 0.90,
                "cutoff_percentile_start": 0.01,
                "cutoff_type": "time",
                "objective_time_keys": [
                    "all"
                ]
            }
        }
    }

    result, env = tune_hyper_params(target_strategy, hyper_params, iterations=1, compiler_options=compiler_options, verbose=True, cache=None)
    assert len(result) == 2
    assert 'best_config' in env
