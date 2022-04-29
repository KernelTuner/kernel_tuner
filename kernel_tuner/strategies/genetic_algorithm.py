""" A simple genetic algorithm for parameter search """

import random
import numpy as np

from kernel_tuner.strategies.minimize import _cost_func
from kernel_tuner.searchspace import Searchspace
from kernel_tuner import util


def tune(runner, kernel_options, device_options, tuning_options):
    """ Find the best performing kernel configuration in the parameter space

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

    options = tuning_options.strategy_options
    pop_size = options.get("popsize", 20)
    generations = options.get("maxiter", 50)
    crossover = supported_methods[options.get("method", "uniform")]
    mutation_chance = options.get("mutation_chance", 10)

    max_fevals = options.get("max_fevals", 100)

    tuning_options["scaling"] = False

    best_time = 1e20
    all_results = []
    unique_results = {}

    searchspace = Searchspace(tuning_options, runner.dev.max_threads)
    population = list(list(p) for p in searchspace.get_random_sample(pop_size))

    # limit max_fevals to max size of the parameter space
    max_fevals = min(searchspace.size, max_fevals)

    for generation in range(generations):

        # determine fitness of population members
        weighted_population = []
        for dna in population:
            time = _cost_func(dna, kernel_options, tuning_options, runner, all_results, check_restrictions=False)
            weighted_population.append((dna, time))

        # population is sorted such that better configs have higher chance of reproducing
        weighted_population.sort(key=lambda x: x[1])

        # 'best_time' is used only for printing
        if tuning_options.verbose and all_results:
            best_time = min(all_results, key=lambda x: x["time"])["time"]

        if tuning_options.verbose:
            print("Generation %d, best_time %f" % (generation, best_time))

        population = []

        unique_results.update({",".join([str(i) for i in dna]): time
                               for dna, time in weighted_population})
        if len(unique_results) >= max_fevals:
            break

        # crossover and mutate
        while len(population) < pop_size:
            dna1, dna2 = weighted_choice(weighted_population, 2)

            children = crossover(dna1, dna2)

            for child in children:
                child = mutate(child, mutation_chance, searchspace)

                if child not in population and searchspace.is_param_config_valid(tuple(child)):
                    population.append(child)

                if len(population) >= pop_size:
                    break

        # TODO could combine old + new generation here and do a selection

    return all_results, runner.dev.get_environment()


def weighted_choice(population, n):
    """Randomly select n unique individuals from a weighted population, fitness determines probability of being selected"""

    def random_index_betavariate(pop_size):
        # has a higher probability of returning index of item at the head of the list
        alpha = 1
        beta = 2.5
        return int(random.betavariate(alpha, beta) * pop_size)

    def random_index_weighted(pop_size):
        """use weights to increase probability of selection"""
        weights = [w for _, w in population]
        # invert because lower is better
        inverted_weights = [1.0 / w for w in weights]
        prefix_sum = np.cumsum(inverted_weights)
        total_weight = sum(inverted_weights)
        randf = random.random() * total_weight
        # return first index of prefix_sum larger than random number
        return next(i for i, v in enumerate(prefix_sum) if v > randf)

    random_index = random_index_betavariate

    indices = [random_index(len(population)) for _ in range(n)]
    chosen = []
    for ind in indices:
        while ind in chosen:
            ind = random_index(len(population))
        chosen.append(ind)

    return [population[ind][0] for ind in chosen]


def mutate(dna, mutation_chance, searchspace: Searchspace):
    """Mutate DNA with 1/mutation_chance chance"""

    # this is actually a neighbors problem with Hamming distance, choose randomly from returned searchspace list
    if int(random.random() * mutation_chance) == 0:
        neighbors = searchspace.get_neighbors(tuple(dna), neighbor_method='Hamming')
        if len(neighbors) > 0:
            return list(random.choice(neighbors))
    return dna


def single_point_crossover(dna1, dna2):
    """crossover dna1 and dna2 at a random index"""
    # check if you can do the crossovers using the neighbor index: check which valid parameter configuration is closest to the crossover, probably best to use "adjacent" as it is least strict?
    pos = int(random.random() * (len(dna1)))
    return (dna1[:pos] + dna2[pos:], dna2[:pos] + dna1[pos:])


def two_point_crossover(dna1, dna2):
    """crossover dna1 and dna2 at 2 random indices"""
    if len(dna1) < 5:
        start, end = 0, len(dna1)
    else:
        start, end = 1, len(dna1) - 1
    pos1, pos2 = sorted(random.sample(list(range(start, end)), 2))
    child1 = dna1[:pos1] + dna2[pos1:pos2] + dna1[pos2:]
    child2 = dna2[:pos1] + dna1[pos1:pos2] + dna2[pos2:]
    return (child1, child2)


def uniform_crossover(dna1, dna2):
    """randomly crossover genes between dna1 and dna2"""
    ind = np.random.random(len(dna1)) > 0.5
    child1 = [dna1[i] if ind[i] else dna2[i] for i in range(len(ind))]
    child2 = [dna2[i] if ind[i] else dna1[i] for i in range(len(ind))]
    return child1, child2


def disruptive_uniform_crossover(dna1, dna2):
    """disruptive uniform crossover

    uniformly crossover genes between dna1 and dna2,
    with children guaranteed to be different from parents,
    if the number of differences between parents is larger than 1
    """
    differences = sum(1 for i, j in zip(dna1, dna2) if i != j)
    swaps = 0
    child1 = dna1[:]
    child2 = dna2[:]
    while swaps < (differences + 1) // 2:
        for ind in range(len(dna1)):
            # if there is a difference on this index and has not been swapped yet
            if dna1[ind] != dna2[ind] and child1[ind] != dna2[ind]:
                p = random.random()
                if p < 0.5 and swaps < (differences + 1) // 2:
                    child1[ind] = dna2[ind]
                    child2[ind] = dna1[ind]
                    swaps += 1
    return (child1, child2)


supported_methods = {
    "single_point": single_point_crossover,
    "two_point": two_point_crossover,
    "uniform": uniform_crossover,
    "disruptive_uniform": disruptive_uniform_crossover
}
