""" The strategy that uses particle swarm optimization"""
import sys
import random
import numpy as np

from kernel_tuner.strategies.minimize import _cost_func
from kernel_tuner.searchspace import Searchspace
from kernel_tuner import util

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
        execution times. And a dictionary that contains information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    results = []

    # SA works with real parameter values and does not need scaling
    tuning_options["scaling"] = False
    args = (kernel_options, tuning_options, runner, results)
    searchspace = Searchspace(tuning_options, runner.dev.max_threads)

    # optimization parameters
    T = tuning_options.strategy_options.get("T", 1.0)
    T_start = T
    T_min = tuning_options.strategy_options.get("T_min", 0.001)
    alpha = tuning_options.strategy_options.get("alpha", 0.995)
    niter = tuning_options.strategy_options.get("maxiter", 1)

    # compute how many iterations would be needed to complete the annealing schedule
    max_iter = int(np.ceil(np.log(T_min)/np.log(alpha)))

    # if user supplied max_fevals that is lower then max_iter we will
    # scale the annealing schedule to fit max_fevals
    max_feval = tuning_options.strategy_options.get("max_fevals", max_iter)

    # get random starting point and evaluate cost
    pos = list(searchspace.get_random_sample(1)[0])
    old_cost = _cost_func(pos, *args, check_restrictions=False)

    # main optimization loop
    stuck = 0
    iteration = 0
    c = 0
    c_old = 0

    while T > T_min:
        if tuning_options.verbose:
            print("iteration: ", iteration, "T", T, "cost: ", old_cost)
            iteration += 1

        for _ in range(niter):

            new_pos = neighbor(pos, searchspace)
            try:
                new_cost = _cost_func(new_pos, *args, check_restrictions=False)
            except util.StopCriterionReached as e:
                if tuning_options.verbose:
                    print(e)
                return results, runner.dev.get_environment()

            ap = acceptance_prob(old_cost, new_cost, T, tuning_options)
            r = random.random()

            if ap > r:
                if tuning_options.verbose:
                    print("new position accepted", new_pos, new_cost, 'old:', pos, old_cost, 'ap', ap, 'r', r, 'T', T)
                pos = new_pos
                old_cost = new_cost

        c = len(tuning_options.unique_results)
        T = T_start * alpha**(max_iter/max_feval*c)

        # check if solver gets stuck and if so restart from random position
        if c == c_old:
            stuck += 1
        else:
            stuck = 0
        c_old = c
        if stuck > 100:
            pos = list(searchspace.get_random_sample(1)[0])
            stuck = 0

        # safeguard
        if iteration > 10*max_iter:
            break

    return results, runner.dev.get_environment()

def acceptance_prob(old_cost, new_cost, T, tuning_options):
    """annealing equation, with modifications to work towards a lower value"""
    error_val = sys.float_info.max if not tuning_options.objective_higher_is_better else -sys.float_info.max
    # if start pos is not valid, always move
    if old_cost == error_val:
        return 1.0
    # if we have found a valid ps before, never move to nonvalid pos
    if new_cost == error_val:
        return 0.0
    # always move if new cost is better
    if new_cost < old_cost:
        return 1.0
    # maybe move if old cost is better than new cost depending on T and random value
    if tuning_options.objective_higher_is_better:
        return np.exp(((new_cost-old_cost)/new_cost)/T)
    return np.exp(((old_cost-new_cost)/old_cost)/T)


def neighbor(pos, searchspace: Searchspace):
    """return a random neighbor of pos"""
    # Note: this is not the same as the previous implementation, because it is possible that non-edge parameters remain the same, but suggested configurations will all be within restrictions
    neighbors = searchspace.get_neighbors(tuple(pos), neighbor_method='Hamming') if random.random() < 0.2 else searchspace.get_neighbors(tuple(pos), neighbor_method='strictly-adjacent')
    if len(neighbors) > 0:
        return list(random.choice(neighbors))
    # if there are no neighbors, return a random configuration
    return list(searchspace.get_random_sample(1)[0])
