""" Iterate over a random sample of the parameter space """
from __future__ import print_function

import itertools
import numpy

from kernel_tuner import util
from time import perf_counter


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

    fraction = tuning_options.strategy_options.get("fraction", 0.1)

    parameter_space = util.get_valid_configs(tuning_options, runner.dev.max_threads)

    # reduce parameter space to a random sample using sample_fraction
    parameter_space = numpy.array(parameter_space)
    size = len(parameter_space)
    fraction = int(numpy.ceil(size * fraction))
    sample_indices = numpy.random.choice(range(size), size=fraction, replace=False)
    parameter_space = parameter_space[sample_indices]

    # call the runner
    results, env = runner.run(parameter_space, kernel_options, tuning_options)

    return results, env
