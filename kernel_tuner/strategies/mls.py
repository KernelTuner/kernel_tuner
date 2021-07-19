""" The strategy that uses multi-start local search """
import random

from kernel_tuner import util
from kernel_tuner.strategies.minimize import _cost_func


def tune(runner, kernel_options, device_options, tuning_options):
    """ Find the best performing kernel configuration in the parameter space

    :params runner: A runner from kernel_tuner.runners
    :type runner: kernel_tuner.runner

    :param kernel_options: A dictionary with all options for the kernel.
    :type kernel_options: dict

    :param device_options: A dictionary with all options for the device
        on which the kernel should be tuned.
    :type device_options: dict

    :param tuning_options: A dictionary with all options regarding the tuning
        process.
    :type tuning_options: dict

    :returns: A list of dictionaries for executed kernel configurations and their
        execution times. And a dictionary that contains a information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """
    # MLS works with real parameter values and does not need scaling
    tuning_options["scaling"] = False
    tune_params = tuning_options.tune_params

    options = tuning_options.strategy_options
    max_fevals = options.get("max_fevals", 100)
    fevals = 0
    max_threads = runner.dev.max_threads

    all_results = []
    unique_results = {}

    #while searching
    while fevals < max_fevals:

        #get random starting position that is valid
        pos = [random.choice(v) for v in tune_params.values()]

        #if we have restrictions and config fails restrictions, try again
        #if restrictions and not util.check_restrictions(restrictions, pos, tune_params.keys(), False):
        if not util.config_valid(pos, tuning_options, max_threads):
            continue

        hillclimb(pos, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner)
        fevals = len(unique_results)

    return all_results, runner.dev.get_environment()


def hillclimb(pos, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner):
    """ simple hillclimbing search until max_fevals is reached or no improvement is found """
    tune_params = tuning_options.tune_params
    max_threads = runner.dev.max_threads

    #measure start point time
    time = _cost_func(pos, kernel_options, tuning_options, runner, all_results)

    #starting new hill climbing search, no need to remember past best
    best_global = best = time

    #store the start pos before hill climbing
    start_pos = pos[:]

    found_improved = True
    while found_improved:
        found_improved = False

        current_results = []
        pos = start_pos[:]

        index = 0
        #in each dimension see the possible values
        for values in tune_params.values():

            #for each value in this dimension
            for value in values:
                pos[index] = value

                #check restrictions
                #if restrictions and not util.check_restrictions(restrictions, pos, tune_params.keys(), False):
                #    continue
                if not util.config_valid(pos, tuning_options, max_threads):
                    continue

                #get time for this position
                time = _cost_func(pos, kernel_options, tuning_options, runner, current_results)
                if time < best:
                    best = time
                    best_pos = pos[:]
                    #greedely replace start_pos with pos to continue from this point
                    start_pos = pos[:]

                unique_results.update({",".join([str(v) for k, v in record.items() if k in tune_params]): record["time"]
                                       for record in current_results})
                fevals = len(unique_results)
                if fevals >= max_fevals:
                    all_results += current_results
                    return

            #restore and move to next dimension
            pos[index] = start_pos[index]
            index = index + 1

        #see if there was improvement, update start_pos set found_improved to True
        if best < best_global:
            found_improved = True
            start_pos = best_pos
            best_global = best

        #append current_results to all_results
        all_results += current_results
