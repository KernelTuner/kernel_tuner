""" The strategy that uses the firefly algorithm for optimization"""
from __future__ import print_function
import numpy as np

from kernel_tuner.strategies.minimize import _cost_func, get_bounds_x0_eps
from kernel_tuner.strategies.pso import Particle

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

    #scale variables in x because PSO works with velocities to visit different configurations
    tuning_options["scaling"] = True

    #using this instead of get_bounds because scaling is used
    bounds, _, _ = get_bounds_x0_eps(tuning_options)

    args = (kernel_options, tuning_options, runner, results, cache)

    num_particles = 20
    maxiter = 100

    #parameters needed by the Firefly Algorithm
    B0 = 1.0
    gamma = 1.0
    alpha = 0.20

    best_time_global = 1e20
    best_position_global = []

    # init particle swarm
    swarm = []
    for i in range(0, num_particles):
        swarm.append(Firefly(bounds, args))

    # compute initial intensities
    for j in range(num_particles):
        swarm[j].compute_intensity(_cost_func)

    for c in range(maxiter):
        if tuning_options.verbose:
            print("start iteration ", c, "best time global", best_time_global)

        # compare all to all and compute attractiveness
        for i in range(num_particles):
            for j in range(num_particles):

                if swarm[i].intensity < swarm[j].intensity:
                    dist = swarm[i].distance_to(swarm[j])
                    beta = B0 * np.exp(-gamma * dist * dist)

                    swarm[i].move_towards(swarm[j], beta, alpha)
                    swarm[i].compute_intensity(_cost_func)

                    # update global best if needed, actually only used for printing
                    if swarm[i].time <= best_time_global:
                        best_position_global = swarm[i].position
                        best_time_global = swarm[i].time

        swarm.sort(key=lambda x: x.time)

    if tuning_options.verbose:
        print('Final result:')
        print(best_position_global)
        print(best_time_global)

    return results, runner.dev.get_environment()


class Firefly(Particle):
    """Firefly object for use in the Firefly Algorithm"""

    def __init__(self, bounds, args):
        """Create Firefly at random position within bounds"""
        super().__init__(bounds, args)
        self.bounds = bounds
        self.intensity = 1 / self.time

    def distance_to(self, other):
        """Return Euclidian distance between self and other Firefly"""
        return np.linalg.norm(self.position-other.position)

    def compute_intensity(self, _cost_func):
        """Evaluate cost function and compute intensity at this position"""
        self.evaluate(_cost_func)
        self.intensity = 1 / self.time

    def move_towards(self, other, beta, alpha):
        """Move firefly towards another given beta and alpha values"""
        self.position += beta * (other.position - self.position)
        self.position += alpha * (np.random.uniform(-0.5, 0.5, len(self.position)))
        self.position = np.minimum(self.position, [b[1] for b in self.bounds])
        self.position = np.maximum(self.position, [b[0] for b in self.bounds])
