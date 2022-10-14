""" The strategy that uses the firefly algorithm for optimization"""
import sys
from collections import OrderedDict

import numpy as np
from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.minimize import (_cost_func, get_bounds_x0_eps,
                                              scale_from_params)
from kernel_tuner.strategies.pso import Particle

_options = OrderedDict(popsize=("Population size", 20),
                       maxiter=("Maximum number of iterations", 100),
                       B0=("Maximum attractiveness", 1.0),
                       gamma=("Light absorption coefficient", 1.0),
                       alpha=("Randomization parameter", 0.2))

def tune(runner, kernel_options, device_options, tuning_options):

    results = []

    # scale variables in x because PSO works with velocities to visit different configurations
    tuning_options["scaling"] = True

    # using this instead of get_bounds because scaling is used
    bounds, _, eps = get_bounds_x0_eps(tuning_options, runner.dev.max_threads)

    args = (kernel_options, tuning_options, runner, results)

    num_particles, maxiter, B0, gamma, alpha = common.get_options(tuning_options.strategy_options, _options)

    best_score_global = sys.float_info.max
    best_position_global = []

    # init particle swarm
    swarm = []
    for i in range(0, num_particles):
        swarm.append(Firefly(bounds, args))

    # ensure particles start from legal points
    searchspace = Searchspace(tuning_options, runner.dev.max_threads)
    population = list(list(p) for p in searchspace.get_random_sample(num_particles))
    for i, particle in enumerate(swarm):
        particle.position = scale_from_params(population[i], tuning_options.tune_params, eps)

    # compute initial intensities
    for j in range(num_particles):
        try:
            swarm[j].compute_intensity(_cost_func)
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)
            return results, runner.dev.get_environment()
        if swarm[j].score <= best_score_global:
            best_position_global = swarm[j].position
            best_score_global = swarm[j].score

    for c in range(maxiter):
        if tuning_options.verbose:
            print("start iteration ", c, "best score global", best_score_global)

        # compare all to all and compute attractiveness
        for i in range(num_particles):
            for j in range(num_particles):

                if swarm[i].intensity < swarm[j].intensity:
                    dist = swarm[i].distance_to(swarm[j])
                    beta = B0 * np.exp(-gamma * dist * dist)

                    swarm[i].move_towards(swarm[j], beta, alpha)
                    try:
                        swarm[i].compute_intensity(_cost_func)
                    except util.StopCriterionReached as e:
                        if tuning_options.verbose:
                            print(e)
                        return results, runner.dev.get_environment()

                    # update global best if needed, actually only used for printing
                    if swarm[i].score <= best_score_global:
                        best_position_global = swarm[i].position
                        best_score_global = swarm[i].score

        swarm.sort(key=lambda x: x.score)

    if tuning_options.verbose:
        print('Final result:')
        print(best_position_global)
        print(best_score_global)

    return results, runner.dev.get_environment()


tune.__doc__ = common.get_strategy_docstring("firefly algorithm", _options)

class Firefly(Particle):
    """Firefly object for use in the Firefly Algorithm"""

    def __init__(self, bounds, args):
        """Create Firefly at random position within bounds"""
        super().__init__(bounds, args)
        self.bounds = bounds
        self.intensity = 1 / self.score

    def distance_to(self, other):
        """Return Euclidian distance between self and other Firefly"""
        return np.linalg.norm(self.position-other.position)

    def compute_intensity(self, _cost_func):
        """Evaluate cost function and compute intensity at this position"""
        self.evaluate(_cost_func)
        if self.score == sys.float_info.max:
            self.intensity = -sys.float_info.max
        else:
            self.intensity = 1 / self.score

    def move_towards(self, other, beta, alpha):
        """Move firefly towards another given beta and alpha values"""
        self.position += beta * (other.position - self.position)
        self.position += alpha * (np.random.uniform(-0.5, 0.5, len(self.position)))
        self.position = np.minimum(self.position, [b[1] for b in self.bounds])
        self.position = np.maximum(self.position, [b[0] for b in self.bounds])
