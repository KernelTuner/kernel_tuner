import random

from kernel_tuner import util
from kernel_tuner.strategies.minimize import _cost_func

def greedy_hillclimb(base_sol, restart, neighbor_method, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner):
    """ Hillclimbing search until max_fevals is reached or no improvement is found.
        Greedy hillclimbing evaluates all neighbouring solutions in a random order
        and immediately moves to the neighbour if it is an improvement.
    """
    if neighbor_method not in ["Hamming", "adjacent"]:
        raise Exception("Unknown neighbour method.")

    tune_params = tuning_options.tune_params
    max_threads = runner.dev.max_threads

    #measure start point time
    best_time = _cost_func(base_sol, kernel_options, tuning_options, runner, all_results)

    found_improved = True
    while found_improved:
        child = base_sol[:]
        found_improved = False

        vals = list(tune_params.values())
        indices = list(range(len(vals)))
        random.shuffle(indices)
        current_results = []
        #in each dimension see the possible values
        for index in indices:
            values = vals[index]
            # If Hamming neighbors, all values are possible neighbors
            if neighbor_method == "Hamming":
                neighbors = values
                random.shuffle(neighbors)
            # If adjacent neighbors, figure out what the adjacent values
            #  are in the list. Those are the only neighbors
            elif neighbor_method == "adjacent":
                which_index = [i for i, x in enumerate(values)
                            if x == child[index]]
                if len(which_index) != 1:
                    raise Exception("Not one unique matching index variable value found among list of possible values.")
                var_idx = which_index[0]
                if var_idx == 0:
                    neighbors = [values[1]]
                elif var_idx == len(values) - 1:
                    neighbors = [values[len(values) - 2]]
                else:
                    neighbors = [values[var_idx - 1], values[var_idx + 1]]

            #for each value in this dimension
            for val in neighbors:
                orig_val = child[index]
                child[index] = val

                #check restrictions
                if not util.config_valid(child, tuning_options, max_threads):
                    child[index] = orig_val
                    continue

                #get time for this position
                time = _cost_func(child, kernel_options, tuning_options, runner, current_results)
                unique_results.update({",".join([str(v) for k, v in record.items() if k in tune_params]): record["time"] for record in current_results})

                #TODO: Wat is Ben's methode om maximalisatie problemen te doen
                if time < best_time:
                    best_time = time
                    base_sol = child[:]
                    found_improved = True
                    if restart:
                        break
                else:
                    child[index] = orig_val

                fevals = len(unique_results)
                if fevals >= max_fevals:
                    all_results += current_results
                    return base_sol
            if found_improved and restart:
                break

        #append current_results to all_results
        all_results += current_results
    return base_sol


def best_improvement_hillclimb(pos, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner):
    """ Hillclimbing search until max_fevals is reached or no improvement is found.
        Best-improvement hillclimbing evaluates all neighbouring solutions and moves
        to the best one every iteration.
    """
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

def ordered_greedy_hillclimb(base_sol, order, restart, neighbor_method, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner):
    """ Hillclimbing search until max_fevals is reached or no improvement is found.
        Ordered greedy hillclimbing evaluates all neighbouring solutions in a prescribed
        order and immediately moves to the neighbour if it is an improvement.
    """
    if neighbor_method not in ["Hamming", "adjacent"]:
        raise Exception("Unknown neighbour method.")

    tune_params = tuning_options.tune_params
    max_threads = runner.dev.max_threads

    #measure start point time
    best_time = _cost_func(base_sol, kernel_options, tuning_options, runner, all_results)

    found_improved = True
    vals = list(tune_params.values())
    while found_improved:
        child = base_sol[:]
        found_improved = False

        if order is None:
            indices = list(range(len(vals)))
        else:
            indices = order

        current_results = []
        #in each dimension see the possible values
        for index in indices:
            values = vals[index]
            # If Hamming neighbors, all values are possible neighbors
            if neighbor_method == "Hamming":
                neighbors = values
            # If adjacent neighbors, figure out what the adjacent values
            #  are in the list. Those are the only neighbors
            elif neighbor_method == "adjacent":
                which_index = [i for i, x in enumerate(values)
                            if x == child[index]]
                if len(which_index) != 1:
                    raise Exception("Not one unique matching index variable value found among list of possible values.")
                var_idx = which_index[0]
                if var_idx == 0:
                    neighbors = [values[1]]
                elif var_idx == len(values) - 1:
                    neighbors = [values[len(values) - 2]]
                else:
                    neighbors = [values[var_idx - 1], values[var_idx + 1]]

            #for each value in this dimension
            for val in neighbors:
                orig_val = child[index]
                child[index] = val

                #check restrictions
                if not util.config_valid(child, tuning_options, max_threads):
                    child[index] = orig_val
                    continue

                #get time for this position
                time = _cost_func(child, kernel_options, tuning_options, runner, current_results)
                unique_results.update({",".join([str(v) for k, v in record.items() if k in tune_params]): record["time"] for record in current_results})

                #TODO: Wat is Ben's methode om maximalisatie problemen te doen
                if time < best_time:
                    best_time = time
                    base_sol = child[:]
                    found_improved = True
                    if restart:
                        break
                else:
                    child[index] = orig_val

                fevals = len(unique_results)
                if fevals >= max_fevals:
                    all_results += current_results
                    return
            if found_improved and restart:
                break

        #append current_results to all_results
        all_results += current_results
