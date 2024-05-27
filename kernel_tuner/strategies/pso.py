"""The strategy that uses particle swarm optimization."""
import random
import sys

import numpy as np

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc, scale_from_params

_options = dict(popsize=("Population size", 20),
                       maxiter=("Maximum number of iterations", 100),
                       w=("Inertia weight constant", 0.5),
                       c1=("Cognitive constant", 2.0),
                       c2=("Social constant", 1.0))

def tune(searchspace: Searchspace, runner, tuning_options):

    #scale variables in x because PSO works with velocities to visit different configurations
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=True)

    #using this instead of get_bounds because scaling is used
    bounds, _, eps = cost_func.get_bounds_x0_eps()


    num_particles, maxiter, w, c1, c2 = common.get_options(tuning_options.strategy_options, _options)

    best_score_global = sys.float_info.max
    best_position_global = []

    # init particle swarm
    swarm = []
    for i in range(0, num_particles):
        swarm.append(Particle(bounds))

    # ensure particles start from legal points
    population = list(list(p) for p in searchspace.get_random_sample(num_particles))
    for i, particle in enumerate(swarm):
        particle.position = scale_from_params(population[i], searchspace.tune_params, eps)

    # start optimization
    for i in range(maxiter):
        if tuning_options.verbose:
            print("start iteration ", i, "best time global", best_score_global)

        # evaluate particle positions
        for j in range(num_particles):
            try:
                swarm[j].evaluate(cost_func)
            except util.StopCriterionReached as e:
                if tuning_options.verbose:
                    print(e)
                return cost_func.results

            # update global best if needed
            if swarm[j].score <= best_score_global:
                best_position_global = swarm[j].position
                best_score_global = swarm[j].score

        # update particle velocities and positions
        for j in range(0, num_particles):
            swarm[j].update_velocity(best_position_global, w, c1, c2)
            swarm[j].update_position(bounds)

    if tuning_options.verbose:
        print('Final result:')
        print(best_position_global)
        print(best_score_global)

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("Particle Swarm Optimization (PSO)", _options)

class Particle:
    def __init__(self, bounds):
        self.ndim = len(bounds)

        self.velocity = np.random.uniform(-1, 1, self.ndim)
        self.position = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        self.best_pos = self.position
        self.best_score = sys.float_info.max
        self.score = sys.float_info.max

    def evaluate(self, cost_func):
        self.score = cost_func(self.position)
        # update best_pos if needed
        if self.score < self.best_score:
            self.best_pos = self.position
            self.best_score = self.score

    def update_velocity(self, best_position_global, w, c1, c2):
        r1 = random.random()
        r2 = random.random()
        vc = c1 * r1 * (self.best_pos - self.position)
        vs = c2 * r2 * (best_position_global - self.position)
        self.velocity = w * self.velocity + vc + vs

    def update_position(self, bounds):
        self.position = self.position + self.velocity
        self.position = np.minimum(self.position, [b[1] for b in bounds])
        self.position = np.maximum(self.position, [b[0] for b in bounds])
