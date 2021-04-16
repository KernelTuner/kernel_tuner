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

    results = []

    # MLS works with real parameter values and does not need scaling
    tuning_options["scaling"] = False
    args = (kernel_options, tuning_options, runner, results)
    tune_params = tuning_options.tune_params
    restrictions = tuning_options.restrictions

    options = tuning_options.strategy_options
    max_fevals = options.get("max_fevals", 100)
    fevals = 0

    all_results = []
    unique_results = {}
    best_global = 1e20
    best = 1e20

    #while searching
    while fevals < max_fevals:

        #get random starting position that is valid
        pos = [random.choice(v) for v in tune_params.values()]

        #if we have restrictions and config fails restrictions, try again
        if restrictions and not util.check_restrictions(restrictions, pos, tune_params.keys(), False):
            continue

        #get time for this position
        time = _cost_func(pos, kernel_options, tuning_options, runner, all_results)
        if time < best:
            best = time

        #store the start pos before hill climbing
        start_pos = pos[:]

        best_global = best #starting new hill climbin search, no need to remember past best global

        found_improved = True
        while found_improved:
            found_improved = False

            if fevals >= max_fevals:
                break

            current_results = []
            pos = start_pos[:]

            index = 0
            #in each dimension see the possible values
            for key, values in tune_params.items():

                #for each value in this dimension
                for v in values:
                    pos[index] = v

                    #check restrictions
                    if restrictions and not util.check_restrictions(restrictions, pos, tune_params.keys(), False):
                        continue

                    #get time for this position
                    time = _cost_func(pos, kernel_options, tuning_options, runner, current_results)
                    if time < best:
                        best = time
                        best_pos = pos[:]

                #restore and move to next dimension
                pos[index] = start_pos[index]
                index = index + 1


            #see if there was improvement, update start_pos set found_improved to True
            if best < best_global:
                found_improved = True
                start_pos = best_pos
                best_global = best
                #print("found improved")
                #print(f"{current_results=}")


            #append current_results to all_results
            all_results += current_results
            unique_results.update({",".join([str(v) for k,v in record.items() if k in tune_params]):record["time"] for record in current_results})
            fevals = len(unique_results)
            #print(fevals, start_pos)



    return all_results, runner.dev.get_environment()
