""" The strategy that uses particle swarm optimization"""

from __future__ import print_function
import random
import numpy as np

from kernel_tuner.strategies.minimize import _cost_func
from kernel_tuner.strategies.genetic_algorithm import random_val

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
    cache = {}

    # SA works with real parameter values and does not need scaling
    tuning_options["scaling"] = False
    args = (kernel_options, tuning_options, runner, results, cache)
    tune_params = tuning_options.tune_params

    # optimization parameters
    T = 1.0
    T_min = 0.001
    alpha = 0.9
    niter = 20

    # generate random starting point and evaluate cost
    pos = []
    for i, _ in enumerate(tune_params.keys()):
        pos.append(random_val(i, tune_params))
    old_cost = _cost_func(pos, *args)

    if tuning_options.verbose:
        c = 0
    # main optimization loop
    while T > T_min:
        if tuning_options.verbose:
            print("iteration: ", c, "T", T, "cost: ", old_cost)
            c += 1

        for i in range(niter):

            new_pos = neighbor(pos, tune_params)
            new_cost = _cost_func(new_pos, *args)

            ap = acceptance_prob(old_cost, new_cost, T)
            r = random.random()

            if ap > r:
                if tuning_options.verbose:
                    print("new position accepted", new_pos, new_cost, 'old:', pos, old_cost, 'ap', ap, 'r', r, 'T', T)
                pos = new_pos
                old_cost = new_cost

        T = T * alpha

    return results, runner.dev.get_environment()

def acceptance_prob(old_cost, new_cost, T):
    """annealing equation, with modifications to work towards a lower value"""
    #if start pos is not valid, always move
    if old_cost == 1e20:
        return 1.0
    #if we have found a valid ps before, never move to nonvalid pos
    if new_cost == 1e20:
        return 0.0
    #always move if new cost is better
    if new_cost < old_cost:
        return 1.0
    #maybe move if old cost is better than new cost depending on T and random value
    return np.exp(((old_cost-new_cost)/old_cost)/T)


def neighbor(pos, tune_params):
    """return a random neighbor of pos"""
    size = len(pos)
    pos_out = []
    # random mutation
    # expected value is set that values all dimensions attempt to get mutated
    for i in range(size):
        key = list(tune_params.keys())[i]
        values = tune_params[key]

        if random.random() < 0.2:  #replace with random value
            new_value = random_val(i, tune_params)
        else: #adjacent value
            ind = values.index(pos[i])
            if random.random() > 0.5:
                ind += 1
            else:
                ind -= 1
            ind = min(max(ind, 0), len(values)-1)
            new_value = values[ind]

        pos_out.append(new_value)
    return pos_out
