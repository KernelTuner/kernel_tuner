""" Iterate over a random sample of the parameter space """
from __future__ import print_function
import numpy

from kernel_tuner.searchspace import Searchspace


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

    # create the search space
    searchspace = Searchspace(tuning_options, runner.dev.max_threads)

    # get the samples
    fraction = tuning_options.strategy_options.get("fraction", 0.1)
    num_samples = int(numpy.ceil(searchspace.size * fraction))
    samples = searchspace.get_random_sample(num_samples)

    # call the runner
    results, env = runner.run(samples, kernel_options, tuning_options)

    return results, env
