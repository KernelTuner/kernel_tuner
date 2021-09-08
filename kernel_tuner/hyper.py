""" Module for functions related to hyperparameter optimization """

import itertools
import numpy as np
import kernel_tuner
import warnings

from kernel_tuner.util import get_config_string


def tune_hyper_params(target_strategy, hyper_params, *args, **kwargs):
    """ Tune hyperparameters for a given strategy and kernel

    This function is to be called just like tune_kernel, except that you specify a strategy
    and a dictionary with hyperparameters in front of the arguments you pass to tune_kernel.

    The arguments to tune_kernel should contain a cachefile. To compute the optimum the hyperparameter
    tuner first tunes the kernel with a brute force search. If your cachefile is not yet complete
    this may take very long.

    :param target_strategy: Specify the strategy for which to tune hyperparameters
    :type target_strategy: string

    :param hyper_params: A dictionary containing the hyperparameters as keys and
        lists the possible values per key
    :type hyper_params: dict(string: list)

    :param args: all positional arguments used to call tune_kernel
    :type args: various

    :param kwargs: other keyword arguments to pass to tune_kernel
    :type kwargs: dict

    """
    if not "cache" in kwargs:
        raise ValueError("Please specify a cachefile to store benchmarking data when tuning hyperparameters")

    def put_if_not_present(d, key, value):
        d[key] = value if not key in d else d[key]

    put_if_not_present(kwargs, "verbose", False)
    put_if_not_present(kwargs, "quiet", True)
    put_if_not_present(kwargs, "simulation_mode", True)
    kwargs['strategy'] = 'brute_force'

    #last position argument is tune_params
    tune_params = args[-1]

    #find optimum
    kwargs["strategy"] = "brute_force"
    results, env = kernel_tuner.tune_kernel(*args, **kwargs)
    optimum = min(results, key=lambda p: p["time"])["time"]

    #could throw a warning for the kwargs that will be overwritten, strategy(_options)
    kwargs["strategy"] = target_strategy

    parameter_space = itertools.product(*hyper_params.values())
    all_results = []

    for params in parameter_space:
        strategy_options = dict(zip(hyper_params.keys(), params))

        kwargs["strategy_options"] = strategy_options

        fevals = []
        p_of_opt = []
        for _ in range(100):
            #measure
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results, env = kernel_tuner.tune_kernel(*args, **kwargs)

            #get unique function evaluations
            unique_fevals = {",".join([str(v) for k, v in record.items() if k in tune_params])
                             for record in results}

            fevals.append(len(unique_fevals))
            # p_of_opt.append(optimum / min(results, key=lambda p: p["time"])["time"] * 100)
            p_of_opt.append(min(results, key=lambda p: p["time"])["time"] / optimum * 100)

        strategy_options["fevals"] = np.average(fevals)
        strategy_options["fevals_std"] = np.std(fevals)

        strategy_options["p_of_opt"] = np.average(p_of_opt)
        strategy_options["p_of_opt_std"] = np.std(p_of_opt)

        print(get_config_string(strategy_options))
        all_results.append(strategy_options)

    return all_results
