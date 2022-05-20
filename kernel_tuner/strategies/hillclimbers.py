import random

from kernel_tuner import util
from kernel_tuner.strategies.minimize import _cost_func
from kernel_tuner.searchspace import Searchspace


def base_hillclimb(base_sol: tuple, neighbor_method: str, max_fevals: int, searchspace: Searchspace, all_results, kernel_options, tuning_options, runner, restart=True, randomize=True, order=None):
    """ Hillclimbing search until max_fevals is reached or no improvement is found

    Base hillclimber that evaluates neighbouring solutions in a random or fixed order
    and possibly immediately moves to the neighbour if it is an improvement.

    :params base_sol: Starting position for hillclimbing
    :type base_sol: list

    :params neighbor_method: Method to use to select neighboring parameter configurations to visit
        during hillclimbing, either "Hamming", "strictly-adjacent" or "adjacent" are supported.
    :type neighbor_method: string

    :params max_fevals: Maximum number of unique function evaluations that is allowed
         during the search.
    :type max_fevals: int

    :params searchspace: The searchspace object.
    :type searchspace: Seachspace

    :params all_results: List of dictionaries with all benchmarked configurations
    :type all_results: list(dict)

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
    if randomize and order:
        raise ValueError("Using a preset order and randomize at the same time is not supported.")

    tune_params = tuning_options.tune_params

    # measure start point score
    best_score = _cost_func(base_sol, kernel_options, tuning_options, runner, all_results, check_restrictions=False)

    found_improved = True
    while found_improved:
        child = list(base_sol[:])
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
            neighbors = searchspace.get_param_neighbors(tuple(child), index, neighbor_method, randomize)

            # for each value in this dimension
            for val in neighbors:
                orig_val = child[index]
                child[index] = val

                # get score for this position
                score = _cost_func(child, kernel_options, tuning_options, runner, current_results, check_restrictions=False)

                # generalize this to other tuning objectives
                if score < best_score:
                    best_score = score
                    base_sol = child[:]
                    found_improved = True
                    if restart:
                        break
                else:
                    child[index] = orig_val

                fevals = len(tuning_options.unique_results)
                if fevals >= max_fevals:
                    all_results += current_results
                    return base_sol
            if found_improved and restart:
                break

        # append current_results to all_results
        all_results += current_results
    return base_sol
