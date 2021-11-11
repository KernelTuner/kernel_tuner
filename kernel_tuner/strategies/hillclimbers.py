import random

from kernel_tuner import util
from kernel_tuner.strategies.minimize import _cost_func


def greedy_hillclimb(base_sol, restart, neighbor_method, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner):
    """ Hillclimbing search until max_fevals is reached or no improvement is found.
        Greedy hillclimbing evaluates all neighbouring solutions in a random order
        and immediately moves to the neighbour if it is an improvement.
    """
    return base_hillclimb(base_sol, neighbor_method, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner, restart=True, randomize=True)

def best_improvement_hillclimb(pos, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner):
    """ Hillclimbing search until max_fevals is reached or no improvement is found.
        Best-improvement hillclimbing evaluates all neighbouring solutions and moves
        to the best one every iteration.
    """
    base_hillclimb(pos, "Hamming", max_fevals, all_results, unique_results, kernel_options, tuning_options, runner, restart=True, randomize=False)


def ordered_greedy_hillclimb(base_sol, order, restart, neighbor_method, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner):
    """ Hillclimbing search until max_fevals is reached or no improvement is found.
        Ordered greedy hillclimbing evaluates all neighbouring solutions in a prescribed
        order and immediately moves to the neighbour if it is an improvement.
    """
    return base_hillclimb(base_sol, neighbor_method, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner, restart=True, randomize=False)


def get_neighbors(neighbor_method, values, element, randomize):
    """ get the list of neighboring elements of element in values """
    # If Hamming neighbors, all values are possible neighbors
    if neighbor_method == "Hamming":
        neighbors = values
        if randomize:
            random.shuffle(neighbors)
    # If adjacent neighbors, figure out what the adjacent values
    # are in the list. Those are the only neighbors
    elif neighbor_method == "adjacent":
        var_idx = values.index(element)
        if var_idx == 0:
            neighbors = [values[1]]
        elif var_idx == len(values) - 1:
            neighbors = [values[len(values) - 2]]
        else:
            neighbors = [values[var_idx - 1], values[var_idx + 1]]
    return neighbors


def base_hillclimb(base_sol, neighbor_method, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner, restart=True, randomize=True):
    """ Hillclimbing search until max_fevals is reached or no improvement is found.
        Greedy hillclimbing evaluates all neighbouring solutions in a random order
        and immediately moves to the neighbour if it is an improvement.
    """
    if neighbor_method not in ["Hamming", "adjacent"]:
        raise ValueError("Unknown neighbour method.")

    tune_params = tuning_options.tune_params
    max_threads = runner.dev.max_threads

    # measure start point time
    best_time = _cost_func(base_sol, kernel_options, tuning_options, runner, all_results)

    found_improved = True
    while found_improved:
        child = base_sol[:]
        found_improved = False

        vals = list(tune_params.values())
        indices = list(range(len(vals)))
        if randomize:
            random.shuffle(indices)
        current_results = []
        # in each dimension see the possible values
        for index in indices:
            values = vals[index]

            neighbors = get_neighbors(neighbor_method, values, child[index], randomize)

            # for each value in this dimension
            for val in neighbors:
                orig_val = child[index]
                child[index] = val

                # check restrictions
                if not util.config_valid(child, tuning_options, max_threads):
                    child[index] = orig_val
                    continue

                # get time for this position
                time = _cost_func(child, kernel_options, tuning_options, runner, current_results)
                unique_results.update({",".join([str(v) for k, v in record.items() if k in tune_params]): record["time"] for record in current_results})

                #TODO: generalize this to other tuning objectives
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

        # append current_results to all_results
        all_results += current_results
    return base_sol

