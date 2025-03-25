"""
The ensemble strategy that optimizes the search through the parameter space using a combination of multiple strategies.
"""

import warnings

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import initialize_ray
from kernel_tuner.runners.simulation import SimulationRunner
from kernel_tuner.util import get_num_devices
from kernel_tuner.runners.parallel import ParallelRunner

from kernel_tuner.strategies import (
    basinhopping,
    bayes_opt,
    diff_evo,
    dual_annealing,
    firefly_algorithm,
    genetic_algorithm,
    greedy_ils,
    greedy_mls,
    minimize,
    mls,
    ordered_greedy_mls,
    pso,
    random_sample,
    simulated_annealing,
)

strategy_map = {
    "random_sample": random_sample,
    "minimize": minimize,
    "basinhopping": basinhopping,
    "diff_evo": diff_evo,
    "genetic_algorithm": genetic_algorithm,
    "greedy_mls": greedy_mls,
    "ordered_greedy_mls": ordered_greedy_mls,
    "greedy_ils": greedy_ils,
    "dual_annealing": dual_annealing,
    "mls": mls,
    "pso": pso,
    "simulated_annealing": simulated_annealing,
    "firefly_algorithm": firefly_algorithm,
    "bayes_opt": bayes_opt,
}

_options = dict(
    ensemble=("List of strategies to be used in the ensemble", ["random_sample", "random_sample"]),
    max_fevals=("Maximum number of function evaluations", None),
    num_gpus=("Number of gpus to run the parallel ensemble on", None),
)


def tune(searchspace: Searchspace, runner, tuning_options, cache_manager=None, actors=None):
    clean_up = True if actors is None and cache_manager is None else False
    options = tuning_options.strategy_options
    simulation_mode = True if isinstance(runner, SimulationRunner) else False
    initialize_ray()

    ensemble, max_fevals, num_gpus = common.get_options(tuning_options.strategy_options, _options)
    num_devices = num_gpus if num_gpus is not None else get_num_devices(simulation_mode=simulation_mode)
    ensemble_size = len(ensemble)

    # setup strategy options
    if "bayes_opt" in ensemble:  # All strategies start from a random sample except for BO
        tuning_options.strategy_options["samplingmethod"] = "random"
    tuning_options.strategy_options["max_fevals"] = 100 * ensemble_size if max_fevals is None else max_fevals
    tuning_options.strategy_options["check_and_retrieve"] = True

    # define number of ray actors needed
    if num_devices < ensemble_size:
        warnings.warn(
            "Number of devices is less than the number of strategies in the ensemble. Some strategies will wait until devices are available.",
            UserWarning,
        )
    num_actors = num_devices if ensemble_size > num_devices else ensemble_size

    ensemble = [strategy_map[strategy] for strategy in ensemble]

    parallel_runner = ParallelRunner(
        runner.kernel_source,
        runner.kernel_options,
        runner.device_options,
        runner.iterations,
        runner.observers,
        num_gpus=num_actors,
        cache_manager=cache_manager,
        simulation_mode=simulation_mode,
        actors=actors,
    )

    final_results = parallel_runner.run(tuning_options=tuning_options, ensemble=ensemble, searchspace=searchspace)

    if clean_up:
        parallel_runner.clean_up_ray()

    return final_results


tune.__doc__ = common.get_strategy_docstring("Ensemble", _options)
