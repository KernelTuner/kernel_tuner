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
    tuning_options["scaling"] = False

    tune_params = tuning_options.tune_params

    population = random_population(dna_size, pop_size, tune_params)

    best_time = 1e20
    all_results = []
    cache = {}

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
            ind1 = weighted_choice(weighted_population)
            ind2 = weighted_choice(weighted_population)

            ind1, ind2 = crossover(ind1, ind2)

            population.append(mutate(ind1, dna_size, tune_params))
            population.append(mutate(ind2, dna_size, tune_params))

    return all_results, runner.dev.get_environment()



def weighted_choice(population):
    """Randomly select, fitness determines probability of being selected"""
    random_number = random.betavariate(1, 2.5) #increased probability of selecting members early in the list
    #random_number = random.random()
    ind = int(random_number*len(population))
    ind = min(max(ind, 0), len(population)-1)
    return population[ind][0]

def random_population(dna_size, pop_size, tune_params):
    """create a random population"""
    population = []
    for _ in range(pop_size):
        dna = []
        for i in range(dna_size):
            dna.append(random_val(i, tune_params))
        population.append(dna)
    return population

def random_val(index, tune_params):
    """return a random value for a parameter"""
    key = list(tune_params.keys())[index]
    return random.choice(tune_params[key])

def mutate(dna, dna_size, tune_params):
    """Mutate DNA with 1/mutation_chance chance"""
    dna_out = []
    mutation_chance = 10
    for i in range(dna_size):
        if int(random.random()*mutation_chance) == 1:
            dna_out.append(random_val(i, tune_params))
        else:
            dna_out.append(dna[i])
    return dna_out

def crossover(dna1, dna2):
    """crossover dna1 and dna2 at a random index"""
    pos = int(random.random()*len(dna1))
    if random.random() < 0.5:
        return (dna1[:pos]+dna2[pos:], dna2[:pos]+dna1[pos:])
    else:
        return (dna2[:pos]+dna1[pos:], dna1[:pos]+dna2[pos:])
