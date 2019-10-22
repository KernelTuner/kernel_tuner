""" A simple genetic algorithm for parameter search """
from __future__ import print_function

import random

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
    pop_size = 20
    generations = 100
    crossover = single_point_crossover

    tuning_options["scaling"] = False
    tune_params = tuning_options.tune_params

    best_time = 1e20
    all_results = []
    cache = {}

    population = random_population(pop_size, tune_params)

    for generation in range(generations):
        if tuning_options.verbose:
            print("Generation %d, best_time %f" % (generation, best_time))

        #determine fitness of population members
        weighted_population = []
        for dna in population:
            time = _cost_func(dna, kernel_options, tuning_options, runner, all_results, cache)
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

            dna1, dna2 = crossover(ind1, ind2)

            population.append(mutate(dna1, tune_params))
            population.append(mutate(dna2, tune_params))

    return all_results, runner.dev.get_environment()



def weighted_choice(population, n):
    """Randomly select n unique individuals from a weighted population, fitness determines probability of being selected"""
    def random_index(pop_size):
        alpha = 1
        beta = 2.5
        return int(random.betavariate(alpha, beta)*pop_size)

    indices = [random_index(len(population)) for _ in range(n)]
    chosen = []
    for ind in indices:
        while ind in chosen:
            ind = random_index(len(population))
        chosen.append(ind)

    return [population[ind][0] for ind in chosen]

def random_population(pop_size, tune_params):
    """create a random population"""
    population = []
    for _ in range(pop_size):
        dna = []
        for i in range(len(tune_params)):
            dna.append(random_val(i, tune_params))
        population.append(dna)
    return population

def random_val(index, tune_params):
    """return a random value for a parameter"""
    key = list(tune_params.keys())[index]
    return random.choice(tune_params[key])

def mutate(dna, tune_params):
    """Mutate DNA with 1/mutation_chance chance"""
    dna_out = []
    mutation_chance = 10
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
    pos1 = int(random.random()*len(dna1))
    pos2 = int(random.random()*len(dna1))
    child1 = dna1[pos1:]+dna2[pos1:pos2]+dna1[:pos2]
    child2 = dna2[pos1:]+dna1[pos1:pos2]+dna2[:pos2]
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
    """disruptive random crossover

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


