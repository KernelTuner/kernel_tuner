""" A simple genetic algorithm for parameter search """
from __future__ import print_function

import random
import numpy as np

from kernel_tuner.strategies.minimize import _cost_func

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
    generations = options.get("maxiter", 100)
    crossover = supported_methods[options.get("method", "uniform")]
    mutation_chance = options.get("mutation_chance", 10)

    tuning_options["scaling"] = False
    tune_params = tuning_options.tune_params

    best_time = 1e20
    all_results = []

    population = random_population(pop_size, tune_params)

    for generation in range(generations):

        #optionally enable something to remove duplicates and increase diversity,
        #leads to longer execution times, but might improve robustness
        #population = ensure_diversity(population, pop_size, tune_params)

        if tuning_options.verbose:
            print("Generation %d, best_time %f" % (generation, best_time))

        #determine fitness of population members
        weighted_population = []
        for dna in population:
            time = _cost_func(dna, kernel_options, tuning_options, runner, all_results)
            weighted_population.append((dna, time))
        population = []

        #'best_time' is used only for printing
        if tuning_options.verbose and all_results:
            best_time = min(all_results, key=lambda x: x["time"])["time"]

        #population is sorted such that better configs have higher chance of reproducing
        weighted_population.sort(key=lambda x: x[1])

        #crossover and mutate
        for _ in range(pop_size//2):
            dna1, dna2 = weighted_choice(weighted_population, 2)

            dna1, dna2 = crossover(dna1, dna2)

            population.append(mutate(dna1, tune_params, mutation_chance))
            population.append(mutate(dna2, tune_params, mutation_chance))

    return all_results, runner.dev.get_environment()


def ensure_diversity(population, pop_size, tune_params):
    """filter duplicates and replace with new random individuals"""
    unique_set = []
    unique_population = []
    for i, dna in enumerate(population):
        string_repr = " ".join([str(d) for d in dna])
        if not string_repr in unique_set:
            unique_population.append(dna)
            unique_set.append(string_repr)
    uniques = len(unique_set)
    #could do different things here to increase population
    #e.g. additional crossovers, or new random configs
    return unique_population + random_population(pop_size - uniques, tune_params)


def weighted_choice(population, n):
    """Randomly select n unique individuals from a weighted population, fitness determines probability of being selected"""
    def random_index_betavariate(pop_size):
        #has a higher probability of returning index of item at the head of the list
        alpha = 1
        beta = 2.5
        return int(random.betavariate(alpha, beta)*pop_size)

    def random_index_weighted(pop_size):
        """might lead to problems: if there's only one valid configuration this method only returns that configuration"""
        weights = [w for _, w in population]
        #invert because lower is better
        inverted_weights = [1.0/w for w in weights]
        prefix_sum = np.cumsum(inverted_weights)
        total_weight = sum(inverted_weights)
        randf = random.random()*total_weight
        #return first index of prefix_sum larger than random number
        return next(i for i,v in enumerate(prefix_sum) if v > randf)

    random_index = random_index_betavariate

    indices = [random_index(len(population)) for _ in range(n)]
    chosen = []
    for ind in indices:
        while ind in chosen:
            ind = random_index(len(population))
        chosen.append(ind)

    return [population[ind][0] for ind in chosen]

def random_population(pop_size, tune_params):
    """create a random population of pop_size unique members"""
    population = []
    option_space = np.prod([len(v) for v in tune_params.values()])
    assert pop_size < option_space
    while len(population) < pop_size:
        dna = []
        for i in range(len(tune_params)):
            dna.append(random_val(i, tune_params))
        if not dna in population:
            population.append(dna)
    return population

def random_val(index, tune_params):
    """return a random value for a parameter"""
    key = list(tune_params.keys())[index]
    return random.choice(tune_params[key])

def mutate(dna, tune_params, mutation_chance):
    """Mutate DNA with 1/mutation_chance chance"""
    dna_out = []
    for i in range(len(dna)):
        if int(random.random()*mutation_chance) == 1:
            dna_out.append(random_val(i, tune_params))
        else:
            dna_out.append(dna[i])
    return dna_out

def single_point_crossover(dna1, dna2):
    """crossover dna1 and dna2 at a random index"""
    pos = int(random.random()*len(dna1))
    return (dna1[:pos]+dna2[pos:], dna2[:pos]+dna1[pos:])

def two_point_crossover(dna1, dna2):
    """crossover dna1 and dna2 at 2 random indices"""
    pos1, pos2 = sorted(random.sample(range(len(dna1)), 2))
    child1 = dna1[:pos1]+dna2[pos1:pos2]+dna1[pos2:]
    child2 = dna2[:pos1]+dna1[pos1:pos2]+dna2[pos2:]
    return (child1, child2)

def uniform_crossover(dna1, dna2):
    """randomly crossover genes between dna1 and dna2"""
    child1 = []
    child2 = []
    for ind in range(len(dna1)):
        p = random.random()
        if p < 0.5:
            child1.append(dna1[ind])
            child2.append(dna2[ind])
        else:
            child2.append(dna1[ind])
            child1.append(dna2[ind])
    return (child1, child2)

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
    while swaps < (differences+1)//2:
        for ind in range(len(dna1)):
            #if there is a difference on this index and has not been swapped yet
            if dna1[ind] != dna2[ind] and child1[ind] != dna2[ind]:
                p = random.random()
                if p < 0.5 and swaps < (differences+1)//2:
                    child1[ind] = dna2[ind]
                    child2[ind] = dna1[ind]
                    swaps += 1
    return (child1, child2)


supported_methods = {"single_point": single_point_crossover,
                     "two_point": two_point_crossover,
                     "uniform": uniform_crossover,
                     "disruptive_uniform": disruptive_uniform_crossover}

