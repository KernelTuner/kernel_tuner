""" The default strategy that iterates through the whole parameter space """
from __future__ import print_function

import itertools

from kernel_tuner import util

def tune(runner, kernel_options, device_options, tuning_options):
    """ Tune all instances in the parameter space

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
    restrictions = tuning_options.restrictions
    verbose = tuning_options.verbose

    #compute cartesian product of all tunable parameters
    parameter_space = itertools.product(*tune_params.values())

    #check for search space restrictions
    if restrictions is not None:
        parameter_space = filter(lambda p: util.check_restrictions(restrictions, p, tune_params.keys(), verbose), parameter_space)

    results, env = runner.run(parameter_space, kernel_options, tuning_options)

    return results, env
