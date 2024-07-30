"""Module for functions related to hyperparameter optimization."""

import itertools
import warnings

import numpy as np

import kernel_tuner
from kernel_tuner.util import get_config_string


def tune_hyper_params(target_strategy, hyper_params, *args, **kwargs):
    """Tune hyperparameters for a given strategy and kernel.

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
    # v Have the methodology as a dependency
    # - User inputs:
    #     - a set of bruteforced cachefiles / template experiments file
    #     - an optimization algorithm
    #     - the hyperparameter values to try
    #     - overarching optimization algorithm (meta-strategy)
    # - At each round:
    #     - The meta-strategy selects a hyperparameter configuration to try
    #     - Kernel Tuner generates an experiments file with the hyperparameter configuration
    #     - Kernel Tuner executes this experiments file using the methodology
    #     - The methodology returns the fitness metric
    #     - The fitness metric is fed back into the meta-strategy



    if "cache" in kwargs:
        del kwargs['cache']

    def put_if_not_present(target_dict, key, value):
        target_dict[key] = value if key not in target_dict else target_dict[key]

    put_if_not_present(kwargs, "verbose", False)
    put_if_not_present(kwargs, "quiet", True)
    kwargs['simulation_mode'] = False
    kwargs['strategy'] = 'dual_annealing'
    kwargs['verify'] = None

    return kernel_tuner.tune_kernel('hyperparamtuning', None, [], [], hyper_params, lang='Hypertuner', *args, **kwargs)




    #last position argument is tune_params
    tune_params = args[-1]

    #find optimum
    kwargs["strategy"] = "brute_force"
    results, _ = kernel_tuner.tune_kernel(*args, **kwargs)
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
                results, _ = kernel_tuner.tune_kernel(*args, **kwargs)

            #get unique function evaluations
            unique_fevals = {",".join([str(v) for k, v in record.items() if k in tune_params])
                             for record in results}

            fevals.append(len(unique_fevals))
            p_of_opt.append(min(results, key=lambda p: p["time"])["time"] / optimum * 100)

        strategy_options["fevals"] = np.average(fevals)
        strategy_options["fevals_std"] = np.std(fevals)

        strategy_options["p_of_opt"] = np.average(p_of_opt)
        strategy_options["p_of_opt_std"] = np.std(p_of_opt)

        print(get_config_string(strategy_options))
        all_results.append(strategy_options)

    return all_results

if __name__ == "__main__":  # TODO remove in production
    hyperparams = {
        'popsize': [10, 20, 30],
        'maxiter': [50, 100, 150],
        'w': [0.25, 0.5, 0.75],
        'c1': [1.0, 2.0, 3.0],
        'c2': [0.5, 1.0, 1.5]
    }
    tune_hyper_params('pso', hyperparams)
