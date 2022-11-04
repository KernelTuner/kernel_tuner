import random

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc


def base_hillclimb(base_sol: tuple, neighbor_method: str, max_fevals: int, searchspace: Searchspace, tuning_options,
                   cost_func: CostFunc, restart=True, randomize=True, order=None):
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

    :param tuning_options: A dictionary with all options regarding the tuning
        process.
    :type tuning_options: dict

    :param cost_func: An instance of `kernel_tuner.strategies.common.CostFunc`
    :type runner: kernel_tuner.strategies.common.CostFunc

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

    tune_params = searchspace.tune_params

    # measure start point score
    best_score = cost_func(base_sol, check_restrictions=False)

    found_improved = True
    while found_improved:
        child = list(base_sol[:])
        found_improved = False

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
                score = cost_func(child, check_restrictions=False)

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
                    return base_sol

            if found_improved and restart:
                break

    return base_sol
