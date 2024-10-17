"""Module for functions related to hyperparameter optimization."""



import kernel_tuner


def tune_hyper_params(target_strategy: str, hyper_params: dict, *args, **kwargs):
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

    iterations = 1
    if "iterations" in kwargs:
        iterations = kwargs['iterations']
        del kwargs['iterations']
    if "cache" in kwargs:
        del kwargs['cache']

    def put_if_not_present(target_dict, key, value):
        target_dict[key] = value if key not in target_dict else target_dict[key]

    put_if_not_present(kwargs, "verbose", True)
    put_if_not_present(kwargs, "quiet", False)
    kwargs['simulation_mode'] = False
    kwargs['strategy'] = 'dual_annealing'
    kwargs['verify'] = None
    arguments = [target_strategy]

    return kernel_tuner.tune_kernel('hyperparamtuning', None, [], arguments, hyper_params, *args, lang='Hypertuner',
                                    objective='score', objective_higher_is_better=True, iterations=iterations, **kwargs)

if __name__ == "__main__":  # TODO remove in production
    # hyperparams = {
    #     'popsize': [10, 20, 30],
    #     'maxiter': [50, 100, 150],
    #     'w': [0.25, 0.5, 0.75],
    #     'c1': [1.0, 2.0, 3.0],
    #     'c2': [0.5, 1.0, 1.5]
    # }
    hyperparams = {
        'popsize': [10],
        'maxiter': [50],
        'w': [0.25, 0.5],
        'c1': [1.0],
        'c2': [0.5]
    }
    result, env = tune_hyper_params('pso', hyperparams)
    print(result)
    print(env['best_config'])
