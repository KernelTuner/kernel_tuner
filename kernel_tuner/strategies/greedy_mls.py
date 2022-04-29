""" A greedy multi-start local search algorithm for parameter search """

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.hillclimbers import base_hillclimb

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

    # retrieve options with defaults
    options = tuning_options.strategy_options
    neighbor = options.get("neighbor", "Hamming")
    restart = options.get("restart", True)
    order = options.get("order", None)
    randomize = options.get("randomize", True)
    max_fevals = options.get("max_fevals", 100)

    tuning_options["scaling"] = False

    # limit max_fevals to max size of the parameter space
    searchspace = Searchspace(tuning_options, runner.dev.max_threads)
    max_fevals = min(searchspace.size, max_fevals)

    fevals = 0
    all_results = []
    unique_results = {}

    #while searching
    while fevals < max_fevals:
        candidate = searchspace.get_random_sample(1)[0]

        base_hillclimb(candidate, neighbor, max_fevals, searchspace, all_results, unique_results, kernel_options, tuning_options, runner, restart=restart, randomize=randomize, order=order)
        fevals = len(unique_results)

    return all_results, runner.dev.get_environment()
