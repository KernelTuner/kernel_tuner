""" Iterate over a random sample of the parameter space """
import numpy as np

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.minimize import _cost_func
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
        execution times. And a dictionary that contains information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    tuning_options["scaling"] = False

    # create the search space
    searchspace = Searchspace(tuning_options, runner.dev.max_threads)

    # get the samples
    fraction = tuning_options.strategy_options.get("fraction", 0.1)
    num_samples = int(np.ceil(searchspace.size * fraction))

    # override if max_fevals is specified
    if "max_fevals" in tuning_options:
        num_samples = tuning_options.max_fevals

    samples = searchspace.get_random_sample(num_samples)

    results = []

    for sample in samples:
        try:
            _cost_func(sample, kernel_options, tuning_options, runner, results, check_restrictions=False)
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)
            return results, runner.dev.get_environment()

    return results, runner.dev.get_environment()
