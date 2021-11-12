""" The strategy that uses multi-start local search """
import itertools
import random

from kernel_tuner import util
from kernel_tuner.strategies.minimize import _cost_func
from kernel_tuner.strategies.greedy_mls import tune as mls_tune

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

    # Default MLS uses 'best improvement' hillclimbing, so disable greedy hillclimbing
    options = tuning_options.strategy_options
    options["restart"] = False
    options["neighbor"] = "Hamming"
    return mls_tune(runner, kernel_options, device_options, tuning_options)
