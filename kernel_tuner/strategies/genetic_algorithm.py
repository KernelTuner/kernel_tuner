""" A simple genetic algorithm for parameter search """
from __future__ import print_function

import random
import numpy as np

from kernel_tuner.strategies.minimize import _cost_func
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
        execution times. And a dictionary that contains a information
        about the hardware/software environment on which the tuning took place.
    :rtype: list(dict()), dict()

    """

    dna_size = len(tuning_options.tune_params.keys())

    options = tuning_options.strategy_options
    pop_size = options.get("popsize", 20)
    generations = options.get("maxiter", 50)
    crossover = supported_methods[options.get("method", "uniform")]
    mutation_chance = options.get("mutation_chance", 10)

    max_fevals = options.get("max_fevals", 100)

    tuning_options["scaling"] = False
    tune_params = tuning_options.tune_params

    best_time = 1e20
    all_results = []
    unique_results = {}

    population = random_population(pop_size, tune_params, tuning_options.restrictions)

    for generation in range(generations):

        if tuning_options.verbose:
            print("Generation %d, best_time %f" % (generation, best_time))
            for dna in population:
                print(dna)
            diversity = len(population)
            for dna1 in population:
                for dna2 in population:
                    if dna1 == dna2:
                        diversity = diversity - 1
            print(f"diversity {diversity}")

        # determine fitness of population members
        weighted_population = []
        for dna in population:
            time = _cost_func(dna, kernel_options, tuning_options, runner, all_results)
            weighted_population.append((dna, time))
        population = []

        # 'best_time' is used only for printing
        if tuning_options.verbose and all_results:
            best_time = min(all_results, key=lambda x: x["time"])["time"]

        unique_results.update({",".join([str(i) for i in dna]): time for dna, time in weighted_population})
        if len(unique_results) > max_fevals:
            break

        # population is sorted such that better configs have higher chance of reproducing
        weighted_population.sort(key=lambda x: x[1])

        # crossover and mutate
        #set1 = weighted_choice(weighted_population, pop_size//2)
        #set2 = weighted_choice(weighted_population, pop_size//2)
        #for dna1, dna2 in zip(set1, set2):
        for _ in range(pop_size//2):
            dna1, dna2 = weighted_choice(weighted_population, 2)

            children = crossover(dna1, dna2)

            for child in children:
                child = mutate(child, tune_params, mutation_chance, tuning_options.restrictions)
                #if child in population:
                #    child = mutate(child, tune_params, 1.0, tuning_options.restrictions)
                population.append(child)


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


def random_population(pop_size, tune_params, restrictions):
    """create a random population of pop_size unique members"""
    population = []
    option_space = np.prod([len(v) for v in tune_params.values()])
    assert pop_size < option_space
    while len(population) < pop_size:
        dna = [random.choice(v) for v in tune_params.values()]
        legal = True
        if restrictions:
            legal = util.check_restrictions(restrictions, dna, tune_params.keys(), False)
        if not dna in population and legal:
            population.append(dna)
    return population


def random_val(index, tune_params):
    """return a random value for a parameter"""
    key = list(tune_params.keys())[index]
    return random.choice(tune_params[key])


def mutate(dna, tune_params, mutation_chance, restrictions):
    """Mutate DNA with 1/mutation_chance chance"""
    dna_out = dna[:]
    if int(random.random() * mutation_chance) == 0:
        attempts = 20
        while attempts > 0:
            #decide which parameter to mutate
            i = random.choice(range(len(dna)))
            dna_out = dna[:]
            dna_out[i] = random_val(i, tune_params)

            legal = True
            if restrictions:
                legal = util.check_restrictions(restrictions, dna_out, tune_params.keys(), False)
            if not dna_out == dna and legal:
                return dna_out
            attempts = attempts - 1
    return dna


def single_point_crossover(dna1, dna2):
    """crossover dna1 and dna2 at a random index"""
    pos = int(random.random() * len(dna1))
    return (dna1[:pos] + dna2[pos:], dna2[:pos] + dna1[pos:])


def two_point_crossover(dna1, dna2):
    """crossover dna1 and dna2 at 2 random indices"""
    pos1, pos2 = sorted(random.sample(range(len(dna1)), 2))
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

