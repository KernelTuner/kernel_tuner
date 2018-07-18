""" The strategy that uses particle swarm optimization"""

from __future__ import print_function
from __future__ import division
import random
import numpy as np

from kernel_tuner.strategies.minimize import _cost_func, get_bounds_x0_eps

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

    best_time_global = 1e20
    best_position_global = []

    # init particle swarm
    swarm = []
    for i in range(0, num_particles):
        swarm.append(Particle(bounds, args))

    for i in range(maxiter):
        if tuning_options.verbose:
            print("start iteration ", i, "best time global", best_time_global)

        # evaluate particle positions
        for j in range(num_particles):
            swarm[j].evaluate(_cost_func)

            # update global best if needed
            if swarm[j].time <= best_time_global:
                best_position_global = swarm[j].position
                best_time_global = swarm[j].time

        # update particle velocities and positions
        for j in range(0, num_particles):
            swarm[j].update_velocity(best_position_global)
            swarm[j].update_position(bounds)

    if tuning_options.verbose:
        print('Final result:')
        print(best_position_global)
        print(best_time_global)

    return results, runner.dev.get_environment()


class Particle:
    def __init__(self, bounds, args):
        self.ndim = len(bounds)
        self.args = args

        self.velocity = np.random.uniform(-1, 1, self.ndim)
        self.position = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        self.best_pos = self.position
        self.best_time = 1e20
        self.time = 1e20

    def evaluate(self, cost_func):
        self.time = cost_func(self.position, *self.args)
        # update best_pos if needed
        if self.time < self.best_time:
            self.best_pos = self.position
            self.best_time = self.time

    def update_velocity(self, best_position_global):
        w = 0.5       # inertia constant
        c1 = 2        # cognitive constant
        c2 = 1        # social constant
        r1 = random.random()
        r2 = random.random()
        vc = c1 * r1 * (self.best_pos - self.position)
        vs = c2 * r2 * (best_position_global - self.position)
        self.velocity = w * self.velocity + vc + vs

    def update_position(self, bounds):
        self.position = self.position + self.velocity
        self.position = np.minimum(self.position, [b[1] for b in bounds])
        self.position = np.maximum(self.position, [b[0] for b in bounds])
