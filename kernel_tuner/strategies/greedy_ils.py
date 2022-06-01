""" A simple greedy iterative local search algorithm for parameter search """

from kernel_tuner.strategies.minimize import _cost_func
from kernel_tuner import util
from kernel_tuner.strategies.hillclimbers import base_hillclimb
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.genetic_algorithm import mutate

def tune(runner, kernel_options, device_options, tuning_options):
    """ Find the best performing kernel configuration in the parameter space

    :params runner: A runner from kernel_tuner.runners
    :type runner: kernel_tuner.runner

    :param kernel_options: A dictionary with all options for the kernel.
    :type kernel_options: kernel_tuner.interface.Options

    :param device_options: A dictionary with all options for the device
        on which the kernel should be tuned.
    :type device_options: kernel_tuner.interface.Options

    :param tuning_options: A dictionary with all options regarding the tuning
        process.
    :type tuning_options: kernel_tuner.interface.Options

    :returns: A list of dictionaries for executed kernel configurations and their
        execution times. And a dictionary that contains information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    dna_size = len(tuning_options.tune_params.keys())

    options = tuning_options.strategy_options

    neighbor = options.get("neighbor", "Hamming")
    restart = options.get("restart", True)
    no_improvement = options.get("no_improvement", 50)
    randomwalk = options.get("random_walk", 0.3)
    perm_size = int(randomwalk * dna_size)
    if perm_size == 0:
        perm_size = 1
    max_fevals = options.get("max_fevals", 100)

    tuning_options["scaling"] = False

    # limit max_fevals to max size of the parameter space
    searchspace = Searchspace(tuning_options, runner.dev.max_threads)
    max_fevals = min(searchspace.size, max_fevals)

    fevals = 0
    results = []

    #while searching
    candidate = searchspace.get_random_sample(1)[0]
    best_score = _cost_func(candidate, kernel_options, tuning_options, runner, results, check_restrictions=False)

    last_improvement = 0
    while fevals < max_fevals:

        try:
            candidate = base_hillclimb(candidate, neighbor, max_fevals, searchspace, results, kernel_options, tuning_options, runner, restart=restart, randomize=True)
            new_score = _cost_func(candidate, kernel_options, tuning_options, runner, results, check_restrictions=False)
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)
            return results, runner.dev.get_environment()

        fevals = len(tuning_options.unique_results)
        if new_score < best_score:
            last_improvement = 0
        else:
            last_improvement += 1

        # Instead of full restart, permute the starting candidate
        candidate = random_walk(candidate, perm_size, no_improvement, last_improvement, searchspace)
    return results, runner.dev.get_environment()


def random_walk(indiv, permutation_size, no_improve, last_improve, searchspace: Searchspace):
    if last_improve >= no_improve:
        return searchspace.get_random_sample(1)[0]
    for _ in range(permutation_size):
        indiv = mutate(indiv, 0, searchspace)
    return indiv
