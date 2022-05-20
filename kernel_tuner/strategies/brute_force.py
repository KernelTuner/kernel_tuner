""" The default strategy that iterates through the whole parameter space """
from __future__ import print_function

from kernel_tuner.searchspace import Searchspace


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
        execution times. And a dictionary that contains information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    # create the searchspace
    searchspace = Searchspace(tuning_options, runner.dev.max_threads, sort=True)

    # call the runner
    results, env = runner.run(searchspace.list, kernel_options, tuning_options)

    return results, env
