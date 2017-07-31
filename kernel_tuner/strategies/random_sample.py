""" Iterate over a random sample of the parameter space """
from __future__ import print_function

import itertools
import numpy

from kernel_tuner import util

def tune(runner, kernel_options, device_options, tuning_options):
    """ Tune a random sample of sample_fraction fraction in the parameter space

    :params runner: A runner from kernel_tuner.runners
    :type runner: kernel_tuner.runner

    :param kernel_options: A dictionary with all options for the kernel.
    :type kernel_options: kernel_tuner.interface.Options

    :param device_options: A dictionary with all options for the device
        on which the kernel should be tuned.
    :type device_options: kernel_tuner.interface.Options

    :param tuning_options: A dictionary with all options regarding the tuning
        process.
    :type tuning_options: kernel_tuner.interface.Options

    :returns: A list of dictionaries for executed kernel configurations and their
        execution times. And a dictionary that contains a information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    tune_params = tuning_options.tune_params

    #compute cartesian product of all tunable parameters
    parameter_space = itertools.product(*tune_params.values())

    #check for search space restrictions
    if tuning_options.restrictions is not None:
        parameter_space = filter(lambda p: util.check_restrictions(tuning_options.restrictions, p, tune_params.keys(), tuning_options.verbose), parameter_space)

    #reduce parameter space to a random sample using sample_fraction
    parameter_space = numpy.array(list(parameter_space))
    size = len(parameter_space)
    sample_indices = numpy.random.choice(range(size), size=int(numpy.ceil(size * float(tuning_options.sample_fraction))), replace=False)
    parameter_space = parameter_space[sample_indices]

    #call the runner
    results, env = runner.run(parameter_space, kernel_options, tuning_options)

    return results, env
