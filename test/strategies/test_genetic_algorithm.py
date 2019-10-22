from __future__ import print_function

from collections import OrderedDict
import numpy as np

import kernel_tuner
from kernel_tuner.strategies import genetic_algorithm as ga


tune_params = OrderedDict()
tune_params["x"] = [1, 2, 3]
tune_params["y"] = [4, 5, 6]



def test_random_population():
    dna_size = 2
    pop_size = 5
    pop = ga.random_population(pop_size, tune_params)

    assert len(pop) == pop_size
    assert len(pop[0]) == 2
    assert pop[0][0] in tune_params["x"]
    assert pop[0][1] in tune_params["y"]


def test_random_val():
    val0 = ga.random_val(0, tune_params)
    assert val0 in tune_params["x"]

    val1 = ga.random_val(1, tune_params)
    assert val1 in tune_params["y"]


def test_mutate():
    pop = ga.random_population(1, tune_params)

    mutant = ga.mutate(pop[0], tune_params)
    assert len(pop[0]) == len(mutant)
    assert mutant[0] in tune_params["x"]
    assert mutant[1] in tune_params["y"]


def test_single_point_crossover():
    #crossover currently implements a 1-point crossover, which
    #does not guarantee that the children are actually different
    #from the parents. It's output is also pointlessly randomized.
    #For now just check if it functions properly

    dna1 = ["x", "y", "z"]
    dna2 = ["a", "b", "c"]

    children = ga.single_point_crossover(dna1, dna2)

    assert len(children) == 2
    assert len(children[0]) == 3
    assert len(children[1]) == 3


