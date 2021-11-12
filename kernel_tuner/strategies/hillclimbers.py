import random

from kernel_tuner import util
from kernel_tuner.strategies.minimize import _cost_func


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


def base_hillclimb(base_sol, neighbor_method, max_fevals, all_results, unique_results, kernel_options, tuning_options, runner, restart=True, randomize=True, order=None):
    """ Hillclimbing search until max_fevals is reached or no improvement is found

    Base hillclimber that evaluates neighbouring solutions in a random or fixed order
    and possibly immediately moves to the neighbour if it is an improvement.

    :params base_sol: Starting position for hillclimbing
    :type base_sol: list

    :params neighbor_method: Method to use to select neighboring positions to visit
        during hillclimbing, either "Hamming" or "adjacent" are supported.
    :type neighbor_method: string

    :params max_fevals: Maximum number of unique function evaluations that is allowed
         during the search.
    :type max_fevals: int

    :params all_results: List of dictionaries with all benchmarked configurations
    :type all_results: list(dict)

    :params unique_results: Dictionaries that records all unique function evaluations
        that count towards max_fevals.
    :type unique_results: dict

    :param kernel_options: A dictionary with all options for the kernel.
    :type kernel_options: dict

    :param tuning_options: A dictionary with all options regarding the tuning
        process.
    :type tuning_options: dict

    :params runner: A runner from kernel_tuner.runners
    :type runner: kernel_tuner.runner

    :params restart: Boolean that controls whether to greedely restart hillclimbing
        from a new position as soon as an improved position is found. True by default.
    :type restart: bool

    :params randomize: Boolean that controls whether the dimensions of the tunable
        parameters are randomized.
    :type randomize: bool

    :params order: Fixed order among the dimensions of the tunable parameters are
        to be evaluated by the hillclimber.
    :type order: list

    :returns: The final position that was reached when hillclimbing halted.
    :rtype: list

    """
    if neighbor_method not in ["Hamming", "adjacent"]:
        raise ValueError("Unknown neighbour method.")
    if randomize and order:
        raise ValueError("Using a preset order and randomize at the same time is not supported.")

    tune_params = tuning_options.tune_params
    max_threads = runner.dev.max_threads

    # measure start point time
    best_time = _cost_func(base_sol, kernel_options, tuning_options, runner, all_results)

    found_improved = True
    while found_improved:
        child = base_sol[:]
        found_improved = False

        current_results = []

        vals = list(tune_params.values())
        if order is None:
            indices = list(range(len(vals)))
        else:
            indices = order
        if randomize:
            random.shuffle(indices)

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

                # generalize this to other tuning objectives
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
