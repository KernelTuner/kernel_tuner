""" Iterate over a random sample of the parameter space """
from collections import OrderedDict

import numpy as np
from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import _cost_func

_options = OrderedDict(fraction=("Fraction of the search space to cover value in [0, 1]", 0.1))


def tune(searchspace: Searchspace, runner, tuning_options):
    tuning_options["scaling"] = False

    # get the samples
    fraction = common.get_options(tuning_options.strategy_options, _options)[0]
    assert 0 <= fraction <= 1.0
    num_samples = int(np.ceil(searchspace.size * fraction))

    # override if max_fevals is specified
    if "max_fevals" in tuning_options:
        num_samples = tuning_options.max_fevals

    samples = searchspace.get_random_sample(num_samples)

    results = []

    for sample in samples:
        try:
            _cost_func(sample, tuning_options, runner, results, check_restrictions=False)
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)
            return results

    return results


tune.__doc__ = common.get_strategy_docstring("Random Sampling", _options)
