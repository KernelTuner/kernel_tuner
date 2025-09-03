"""Module for functions related to hyperparameter optimization."""


from pathlib import Path
from random import randint
from argparse import ArgumentParser

import numpy as np

import kernel_tuner


def get_random_unique_filename(prefix = '', suffix=''):
    """Get a random, unique filename that does not yet exist."""
    def randpath():
        return Path(f"{prefix}{randint(1000, 9999)}{suffix}")
    path = randpath()
    while path.exists():
        path = randpath()
    return path

def tune_hyper_params(target_strategy: str, hyper_params: dict, restrictions: list, *args, **kwargs):
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

    # pass a temporary cache file to avoid duplicate execution
    if 'cache' not in kwargs:
        cachefile = get_random_unique_filename('temp_', '.json')
        cachefile = Path(f"hyperparamtuning_paper_bruteforce_{target_strategy}.json")
        kwargs['cache'] = str(cachefile)

    def put_if_not_present(target_dict, key, value):
        target_dict[key] = value if key not in target_dict else target_dict[key]

    # set default arguments if not provided
    put_if_not_present(kwargs, "verbose", True)
    put_if_not_present(kwargs, "quiet", False)
    put_if_not_present(kwargs, "simulation_mode", False)
    put_if_not_present(kwargs, "strategy", 'brute_force')
    put_if_not_present(kwargs, 'verify', None)
    arguments = [target_strategy]

    # IMPORTANT when running this script in parallel, always make sure the below name is unique among your runs!
    # e.g. when parallalizing over the hypertuning of multiple strategies, use the strategy name
    name = f"hyperparamtuning_{target_strategy.lower()}"

    # execute the hyperparameter tuning
    result, env = kernel_tuner.tune_kernel(name, None, [], arguments, hyper_params, restrictions=restrictions, *args, lang='Hypertuner',
                                    objective='score', objective_higher_is_better=True, iterations=iterations, **kwargs)
    
    # remove the temporary cachefile and return only unique results in order
    # cachefile.unlink()
    result_unique = dict()
    for r in result:
        config_id = ",".join(str(r[k]) for k in hyper_params.keys())
        if config_id not in result_unique:
            result_unique[config_id] = r
    return list(result_unique.values()), env

if __name__ == "__main__":
    """Main function to run the hyperparameter tuning. Run with `python hyper.py strategy_to_tune=`."""

    parser = ArgumentParser()
    parser.add_argument("strategy_to_tune", type=str, help="The strategy to tune hyperparameters for.")
    parser.add_argument("--meta_strategy", nargs='?', default="dual_annealing", type=str, help="The meta-strategy to use for hyperparameter tuning.")
    parser.add_argument("--max_time", nargs='?', default=60*60*24, type=int, help="The maximum time in seconds for the hyperparameter tuning.")
    args = parser.parse_args()
    strategy_to_tune = args.strategy_to_tune

    kwargs = dict(
        verbose=True,
        quiet=False,
        simulation_mode=False,
        strategy=args.meta_strategy,
        cache=f"hyperparamtuning_t={strategy_to_tune}_m={args.meta_strategy}.json",
        strategy_options=dict(
            time_limit=args.max_time,
        )
    )

    # select the hyperparameter parameters for the selected optimization algorithm
    restrictions = []
    if strategy_to_tune.lower() == "pso":
        # exhaustive search for PSO hyperparameters
        # hyperparams = {
        #     'popsize': [10, 20, 30],
        #     'maxiter': [50, 100, 150],
        #     # 'w': [0.25, 0.5, 0.75],   # disabled due to low influence according to KW-test (H=0.0215) and mutual information
        #     'c1': [1.0, 2.0, 3.0],
        #     'c2': [0.5, 1.0, 1.5]
        # }
        hyperparams = {
            'popsize': list(range(2, 50+1, 2)),
            'maxiter': list(range(10, 200, 10)),
            # 'w': [0.25, 0.5, 0.75],   # disabled due to low influence according to KW-test (H=0.0215) and mutual information
            'c1': [round(n, 2) for n in np.arange(1.0, 3.5+0.25, 0.25).tolist()],
            'c2': [round(n, 2) for n in np.arange(0.5, 2.0+0.25, 0.25).tolist()]
        }
    elif strategy_to_tune.lower() == "firefly_algorithm":
        hyperparams = {
            'popsize': [10, 20, 30],
            'maxiter': [50, 100, 150],
            'B0': [0.5, 1.0, 1.5],
            'gamma': [0.1, 0.25, 0.5],
            'alpha': [0.1, 0.2, 0.3]
        }
    elif strategy_to_tune.lower() == "greedy_ils":
        hyperparams = {
            'neighbor': ['Hamming', 'adjacent'],
            'restart': [True, False],
            'no_improvement': [10, 25, 50, 75],
            'random_walk': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    elif strategy_to_tune.lower() == "dual_annealing":
        hyperparams = {
            'method': ['COBYLA', 'L-BFGS-B', 'SLSQP', 'CG', 'Powell', 'Nelder-Mead', 'BFGS', 'trust-constr'],
        }
    elif strategy_to_tune.lower() == "diff_evo":
        hyperparams = {
            'method': ["best1bin", "rand1bin", "best2bin", "rand2bin", "best1exp", "rand1exp", "best2exp", "rand2exp", "currenttobest1bin", "currenttobest1exp", "randtobest1bin", "randtobest1exp"],   # best1bin
            'popsize': list(range(2, 50+1, 2)),   # 50
            'popsize_times_dimensions': [True, False],  # False
            'F': [round(n, 2) for n in np.arange(0.1, 2.0+0.1, 0.1).tolist()],  # 1.3
            'CR': [round(n, 2) for n in np.arange(0.05, 1.0+0.05, 0.05).tolist()]  # 0.9
        }
    elif strategy_to_tune.lower() == "basinhopping":
        hyperparams = {
            'method': ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"],
            'T': [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        }
    elif strategy_to_tune.lower() == "genetic_algorithm":
        # hyperparams = {
        #     'method': ["single_point", "two_point", "uniform", "disruptive_uniform"],
        #     'popsize': [10, 20, 30],
        #     'maxiter': [50, 100, 150],
        #     'mutation_chance': [5, 10, 20]
        # }
        hyperparams = {
            'method': ["single_point", "two_point", "uniform", "disruptive_uniform"],
            'popsize': list(range(2, 50+1, 2)),
            'maxiter': list(range(10, 200, 10)),
            'mutation_chance': list(range(5, 100, 5))
        }
    elif strategy_to_tune.lower() == "greedy_mls":
        hyperparams = {
            'neighbor': ["Hamming", "adjacent"],
            'restart': [True, False],
            'randomize': [True, False]
        }
    elif strategy_to_tune.lower() == "simulated_annealing":
        # hyperparams = {
        #     'T': [0.5, 1.0, 1.5],
        #     'T_min': [0.0001, 0.001, 0.01],
        #     'alpha': [0.9925, 0.995, 0.9975],
        #     'maxiter': [1, 2, 3]
        # }
        hyperparams = {
            'T': [round(n, 2) for n in np.arange(0.1, 2.0+0.1, 0.1).tolist()],
            'T_min': [round(n, 4) for n in np.arange(0.0001, 0.1, 0.001).tolist()],
            'alpha': [0.9925, 0.995, 0.9975],
            'maxiter': list(range(1, 10, 1))
        }
    elif strategy_to_tune.lower() == "bayes_opt":
        hyperparams = {
            # 'covariancekernel': ["constantrbf", "rbf", "matern32", "matern52"],
            'covariancelengthscale': [1.0, 1.5, 2.0],
            'method': ["poi", "ei", "lcb", "lcb-srinivas", "multi", "multi-advanced", "multi-fast", "multi-ultrafast"],
            'samplingmethod': ["random", "LHS"],
            'popsize': [10, 20, 30]
        }
    else:
        raise ValueError(f"Invalid argument {strategy_to_tune=}")

    # run the hyperparameter tuning
    result, env = tune_hyper_params(strategy_to_tune.lower(), hyperparams, restrictions=restrictions, **kwargs)
    print(result)
    print(env['best_config'])
