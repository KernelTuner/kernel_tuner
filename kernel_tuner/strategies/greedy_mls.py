""" A greedy multi-start local search algorithm for parameter search """

import itertools
import random

from kernel_tuner.strategies.minimize import _cost_func
from kernel_tuner import util
from kernel_tuner.strategies.hillclimbers import greedy_hillclimb
from kernel_tuner.strategies.genetic_algorithm import random_population

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
        execution times. And a dictionary that contains a information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    dna_size = len(tuning_options.tune_params.keys())

    options = tuning_options.strategy_options

    neighbour = options.get("neighbor", "Hamming")
    restart = options.get("restart", True)
    max_fevals = options.get("max_fevals", 100)

    tuning_options["scaling"] = False
    tune_params = tuning_options.tune_params

    # limit max_fevals to max size of the parameter space
    parameter_space = itertools.product(*tune_params.values())
    if tuning_options.restrictions is not None:
        parameter_space = filter(lambda p: util.check_restrictions(tuning_options.restrictions, p, tune_params.keys(), tuning_options.verbose), parameter_space)
    max_elems = len(list(parameter_space))
    if max_elems < max_fevals:
        max_fevals = max_elems

    fevals = 0
    max_threads = runner.dev.max_threads
    all_results = []
    unique_results = {}

    #while searching
    while fevals < max_fevals:
        candidate = random_population(1, tune_params, tuning_options, max_threads)[0]

        _ = greedy_hillclimb(candidate, restart, neighbour, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner)
        fevals = len(unique_results)
    return all_results, runner.dev.get_environment()

