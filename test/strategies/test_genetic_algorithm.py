from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import genetic_algorithm as ga

tune_params = dict()
tune_params["x"] = [1, 2, 3]
tune_params["y"] = [4, 5, 6]

max_threads = 1024
searchspace = Searchspace(tune_params, [], max_threads)


def test_weighted_choice():
    pop_size = 5
    pop = searchspace.get_random_sample(pop_size)
    weighted_pop = [[p, i] for i, p in enumerate(pop)]

    result = ga.weighted_choice(weighted_pop, 1)
    assert result[0] in pop

    result = ga.weighted_choice(weighted_pop, 2)
    print(result)
    assert result[0] in pop
    assert result[1] in pop
    assert result[0] != result[1]


def test_random_population():
    pop_size = 8
    pop = searchspace.get_random_sample(pop_size)

    assert len(pop) == pop_size
    assert len(pop[0]) == 2
    assert pop[0][0] in tune_params["x"]
    assert pop[0][1] in tune_params["y"]

    # test all members are unique
    for i, dna1 in enumerate(pop):
        for j, dna2 in enumerate(pop):
            if i != j:
                assert dna1 != dna2


def test_mutate():
    pop = searchspace.get_random_sample(1)

    mutant = ga.mutate(pop[0], 10, searchspace)
    assert len(pop[0]) == len(mutant)
    assert mutant[0] in tune_params["x"]
    assert mutant[1] in tune_params["y"]


def test_crossover_functions():
    dna1 = ["x", "y", "z"]
    dna2 = ["a", "b", "c"]
    funcs = ga.supported_methods.values()
    for func in funcs:
        children = func(dna1, dna2)
        print(dna1, dna2)
        print(children)
        assert len(children) == 2
        assert len(children[0]) == 3
        assert len(children[1]) == 3


def test_disruptive_uniform_crossover():
    # two individuals with at exactly 2 differences
    dna1 = ["x", "y", 1, 2, 3, 4, 5]
    dna2 = ["x", "x", 1, 2, 3, 4, 7]
    # confirm that disruptive uniform crossover indeed guarantees
    # offsping that is different from the parents and each other
    # when there is more than 1 difference between the parents
    child1, child2 = ga.disruptive_uniform_crossover(dna1, dna2)
    assert child1 != dna1 and child1 != dna2
    assert child2 != dna1 and child2 != dna2
    assert child1 != child2
