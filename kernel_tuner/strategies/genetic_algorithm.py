"""A simple genetic algorithm for parameter search."""

import random

import numpy as np

from kernel_tuner.util import StopCriterionReached, get_best_config
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc

_options = dict(
    popsize=("population size", 26),
    maxiter=("maximum number of generations", 90),
    method=("crossover method to use, choose any from single_point, two_point, uniform, disruptive_uniform", "single_point"),
    mutation_chance=("chance to mutate is 1 in mutation_chance", 55),
    constraint_aware=("constraint-aware optimization (True/False)", True),
)


def tune(searchspace: Searchspace, runner, tuning_options):

    options = tuning_options.strategy_options
    pop_size, generations, method, mutation_chance, constraint_aware = common.get_options(options, _options)

    # if necessary adjust the popsize to a sensible value based on search space size
    if pop_size < 2 or pop_size > np.floor(searchspace.size / 2):
        pop_size = min(max(round((searchspace.size / generations) * 3), 2), pop_size)

    GA = GeneticAlgorithm(pop_size, searchspace, method, mutation_chance, constraint_aware)

    best_score = 1e20
    cost_func = CostFunc(searchspace, tuning_options, runner)
    num_evaluated = 0

    population = GA.generate_population()

    population[0] = cost_func.get_start_pos()

    for generation in range(generations):
        if constraint_aware and any([not searchspace.is_param_config_valid(tuple(dna)) for dna in population]):
            raise ValueError(f"Generation {generation}/{generations}, population validity: {[searchspace.is_param_config_valid(tuple(dna)) for dna in population]}")

        # determine fitness of population members
        weighted_population = []
        for dna in population:
            try:
                # if we are not constraint-aware we should check restrictions upon evaluation
                time = cost_func(dna, check_restrictions=not constraint_aware)
                num_evaluated += 1
            except StopCriterionReached as e:
                if tuning_options.verbose:
                    print(e)
                return cost_func.results

            weighted_population.append((dna, time))

        # population is sorted such that better configs have higher chance of reproducing
        weighted_population.sort(key=lambda x: x[1])

        # 'best_score' is used only for printing
        if tuning_options.verbose and cost_func.results:
            best_score = get_best_config(
                cost_func.results, tuning_options.objective, tuning_options.objective_higher_is_better
            )[tuning_options.objective]

        if tuning_options.verbose:
            print("Generation %d, best_score %f" % (generation, best_score))

        # build new population for next generation
        population = []

        # crossover and mutate
        while len(population) < pop_size and searchspace.size > num_evaluated + len(population):
            dna1, dna2 = GA.weighted_choice(weighted_population, 2)

            children = GA.crossover(dna1, dna2)

            for child in children:
                child = GA.mutate(child)

                if child not in population and (not constraint_aware or searchspace.is_param_config_valid(tuple(child))):
                    population.append(child)

                if len(population) >= pop_size:
                    break

        # could combine old + new generation here and do a selection

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("Genetic Algorithm", _options)

class GeneticAlgorithm:

    def __init__(self, pop_size, searchspace, method="uniform", mutation_chance=10, constraint_aware=True):
        self.pop_size = pop_size
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()
        self.crossover_method = supported_methods[method]
        self.mutation_chance = mutation_chance
        self.constraint_aware = constraint_aware

    def generate_population(self):
        """ Constraint-aware population creation method """
        if self.constraint_aware:
            pop = list(list(p) for p in self.searchspace.get_random_sample(self.pop_size))
        else:
            pop = []
            dna_size = len(self.tune_params)
            for _ in range(self.pop_size):
                dna = []
                for key in self.tune_params:
                    dna.append(random.choice(self.tune_params[key]))
                pop.append(dna)
        return pop

    def crossover(self, dna1, dna2):
        """ Apply selected crossover method, repair dna if constraint-aware """
        dna1, dna2 = self.crossover_method(dna1, dna2)
        if self.constraint_aware:
            return self.repair(dna1), self.repair(dna2)
        return dna1, dna2

    def weighted_choice(self, population, n):
        """Randomly select n unique individuals from a weighted population, fitness determines probability of being selected."""

        def random_index_betavariate(pop_size):
            # has a higher probability of returning index of item at the head of the list
            alpha = 1
            beta = 2.5
            return int(random.betavariate(alpha, beta) * pop_size)

        def random_index_weighted(pop_size):
            """Use weights to increase probability of selection."""
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


    def mutate(self, dna):
        """Mutate DNA with 1/mutation_chance chance."""
        # this is actually a neighbors problem with Hamming distance, choose randomly from returned searchspace list
        if int(random.random() * self.mutation_chance) == 0:
            if self.constraint_aware:
                neighbor = self.searchspace.get_random_neighbor(tuple(dna), neighbor_method="Hamming")
                if neighbor is not None:
                    return list(neighbor)
            else:
                # select a tunable parameter at random
                mutate_index = random.randint(0, len(self.tune_params)-1)
                mutate_key = list(self.tune_params.keys())[mutate_index]
                # get all possible values for this parameter and remove current value
                new_val_options = self.tune_params[mutate_key].copy()
                new_val_options.remove(dna[mutate_index])
                # pick new value at random
                if len(new_val_options) > 0:
                    new_val = random.choice(new_val_options)
                    dna[mutate_index] = new_val
        return dna


    def repair(self, dna):
        """ It is possible that crossover methods yield a configuration that is not valid. """
        if not self.searchspace.is_param_config_valid(tuple(dna)):
            # dna is not valid, try to repair it
            # search for valid configurations neighboring this config
            # start from strictly-adjacent to increasingly allowing more neighbors
            for neighbor_method in ["strictly-adjacent", "adjacent", "Hamming"]:
                neighbor = self.searchspace.get_random_neighbor(tuple(dna), neighbor_method=neighbor_method)
                # if we have found valid neighboring configurations, select one at random
                if neighbor is not None:
                    # print(f"GA crossover resulted in invalid config {dna=}, repaired dna to {neighbor=}")
                    return list(neighbor)

        return dna


def single_point_crossover(dna1, dna2):
    """Crossover dna1 and dna2 at a random index."""
    # check if you can do the crossovers using the neighbor index: check which valid parameter configuration is closest to the crossover, probably best to use "adjacent" as it is least strict?
    pos = int(random.random() * (len(dna1)))
    return dna1[:pos] + dna2[pos:], dna2[:pos] + dna1[pos:]


def two_point_crossover(dna1, dna2):
    """Crossover dna1 and dna2 at 2 random indices."""
    if len(dna1) < 5:
        start, end = 0, len(dna1)
    else:
        start, end = 1, len(dna1) - 1
    pos1, pos2 = sorted(random.sample(list(range(start, end)), 2))
    child1 = dna1[:pos1] + dna2[pos1:pos2] + dna1[pos2:]
    child2 = dna2[:pos1] + dna1[pos1:pos2] + dna2[pos2:]
    return child1, child2


def uniform_crossover(dna1, dna2):
    """Randomly crossover genes between dna1 and dna2."""
    ind = np.random.random(len(dna1)) > 0.5
    child1 = [dna1[i] if ind[i] else dna2[i] for i in range(len(ind))]
    child2 = [dna2[i] if ind[i] else dna1[i] for i in range(len(ind))]
    return child1, child2


def disruptive_uniform_crossover(dna1, dna2):
    """Disruptive uniform crossover.

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
    return child1, child2


supported_methods = {
    "single_point": single_point_crossover,
    "two_point": two_point_crossover,
    "uniform": uniform_crossover,
    "disruptive_uniform": disruptive_uniform_crossover,
}

