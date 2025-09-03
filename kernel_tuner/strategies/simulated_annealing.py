"""The strategy that uses particle swarm optimization."""
import random
import sys

import numpy as np

from kernel_tuner.util import StopCriterionReached, ErrorConfig
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc


_options = dict(T=("Starting temperature", 0.1),
                T_min=("End temperature", 0.0001),
                alpha=("Alpha parameter", 0.9975),
                maxiter=("Number of iterations within each annealing step", 1),
                constraint_aware=("constraint-aware optimization (True/False)", True))

def tune(searchspace: Searchspace, runner, tuning_options):
    # SA works with real parameter values and does not need scaling
    cost_func = CostFunc(searchspace, tuning_options, runner, return_invalid=True)

    # optimization parameters
    T, T_min, alpha, niter, constraint_aware = common.get_options(tuning_options.strategy_options, _options)
    T_start = T

    # compute how many iterations would be needed to complete the annealing schedule
    max_iter = int(np.ceil(np.log(T_min)/np.log(alpha)))

    # if user supplied max_fevals that is lower then max_iter we will
    # scale the annealing schedule to fit max_fevals
    max_fevals = tuning_options.strategy_options.get("max_fevals", max_iter)

    # limit max_fevals to max size of the parameter space
    max_fevals = min(searchspace.size, max_fevals)

    # get random starting point and evaluate cost
    pos = cost_func.get_start_pos()
    old_cost = cost_func(pos, check_restrictions=not constraint_aware)

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

            new_pos = neighbor(pos, searchspace, constraint_aware)
            try:
                new_cost = cost_func(new_pos, check_restrictions=not constraint_aware)
            except StopCriterionReached as e:
                if tuning_options.verbose:
                    print(e)
                return cost_func.results

            ap = acceptance_prob(old_cost, new_cost, T, tuning_options)
            r = random.random()

            if ap > r:
                if tuning_options.verbose:
                    print("new position accepted", new_pos, new_cost, 'old:', pos, old_cost, 'ap', ap, 'r', r, 'T', T)
                pos = new_pos
                old_cost = new_cost

        c = len(tuning_options.unique_results)
        T = T_start * alpha**(max_iter/max_fevals*c)

        # check if solver gets stuck and if so restart from random position
        if c == c_old:
            stuck += 1
        else:
            stuck = 0
        c_old = c
        if stuck > 100:
            pos = generate_starting_point(searchspace, constraint_aware)
            stuck = 0

        # safeguard
        if iteration > 10*max_iter:
            break

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("Simulated Annealing", _options)

def acceptance_prob(old_cost, new_cost, T, tuning_options):
    """Annealing equation, with modifications to work towards a lower value."""
    res = 0.0
    # if start pos is not valid, always move
    if isinstance(old_cost, ErrorConfig):
        res = 1.0
    # if we have found a valid ps before, never move to nonvalid pos
    elif isinstance(new_cost, ErrorConfig):
        res = 0.0
    # always move if new cost is better
    elif new_cost < old_cost:
        res = 1.0
    # maybe move if old cost is better than new cost depending on T and random value
    else:
        if tuning_options.objective_higher_is_better:
            res = np.exp(((new_cost-old_cost)/new_cost)/T)
        else:
            res = np.exp(((old_cost-new_cost)/old_cost)/T)
    return res


def neighbor(pos, searchspace: Searchspace, constraint_aware=True):
    """Return a random neighbor of pos."""

    def random_neighbor(pos, method):
        """Helper method to return a random neighbor."""
        neighbor = searchspace.get_random_neighbor(pos, neighbor_method=method)
        if neighbor is None:
            return pos
        return neighbor

    size = len(pos)

    if constraint_aware:
        pos = tuple(pos)

        # Note: the following tries to mimick as much as possible the earlier version of SA but in a constraint-aware version
        for i in range(size):
            if random.random() < 0.2:
                pos = random_neighbor(pos, 'Hamming')
        pos = random_neighbor(pos, 'adjacent')

        return list(pos)

    else:
        tune_params = searchspace.tune_params
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

def random_val(index, tune_params):
    """return a random value for a parameter"""
    key = list(tune_params.keys())[index]
    return random.choice(tune_params[key])

def generate_starting_point(searchspace: Searchspace, constraint_aware=True):
    if constraint_aware:
        return list(searchspace.get_random_sample(1)[0])
    else:
        tune_params = searchspace.tune_params
        return [random_val(i, tune_params) for i in range(len(tune_params))]
