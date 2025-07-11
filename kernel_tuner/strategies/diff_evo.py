"""A simple Different Evolution for parameter search."""
import random
import re
import numpy as np

from kernel_tuner.util import StopCriterionReached
from scipy.stats.qmc import LatinHypercube
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc

_options = dict(
    popsize=("population size", 50),
    maxiter=("maximum number of generations", 200),
    F=("mutation factor (differential weight)", 1.3),
    CR=("crossover rate", 0.9),
    method=("method", "best1bin"),
    constraint_aware=("constraint-aware optimization (True/False)", True),
)

supported_methods = [
    "best1bin",
    "rand1bin",
    "best2bin",
    "rand2bin",
    "best1exp",
    "rand1exp",
    "best2exp",
    "rand2exp",
    "currenttobest1bin",
    "currenttobest1exp",
    "randtobest1bin",
    "randtobest1exp",
]


def tune(searchspace: Searchspace, runner, tuning_options):
    cost_func = CostFunc(searchspace, tuning_options, runner)
    bounds = cost_func.get_bounds()

    options = tuning_options.strategy_options
    popsize, maxiter, F, CR, method, constraint_aware = common.get_options(options, _options)

    if method not in supported_methods:
        raise ValueError(f"Error {method} not supported, {supported_methods=}")

    try:
        differential_evolution(searchspace, cost_func, bounds, popsize, maxiter, F, CR, method, constraint_aware, tuning_options.verbose)
    except StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("Differential Evolution", _options)


def values_to_indices(individual_values, tune_params):
    """Converts an individual's values to its corresponding index vector."""
    idx = np.zeros(len(individual_values))
    for i, v in enumerate(tune_params.values()):
        idx[i] = v.index(individual_values[i])
    return idx


def indices_to_values(individual_indices, tune_params):
    """Converts an individual's index vector back to its values."""
    tune_params_list = list(tune_params.values())
    values = []
    for dim, idx in enumerate(individual_indices):
        values.append(tune_params_list[dim][idx])
    return values


def parse_method(method):
    """Helper func to parse the preferred method into its components."""
    pattern = r"^(best|rand|currenttobest|randtobest)(1|2)(bin|exp)$"
    match = re.fullmatch(pattern, method)

    if match:
        if match.group(1) in ["currenttobest", "randtobest"]:
            mutation_method = mutation[match.group(1)]
        else:
            mutation_method = mutation[match.group(2)]
        return match.group(1) == "best", int(match.group(2)), mutation_method, crossover[match.group(3)]
    else:
        raise ValueError("Error parsing differential evolution method")


def random_draw(idxs, mutate, best):
    """
    Draw requested number of random individuals.

    Draw without replacement unless there is not enough to draw from.
    """
    draw = 2 * mutate + 1 - int(best)
    return np.random.choice(idxs, draw, replace=draw >= len(idxs))


def generate_population(tune_params, max_idx, popsize, searchspace, constraint_aware):
    """ Generate new population, returns Numpy array """
    if constraint_aware:
        samples = LatinHypercube(len(tune_params)).integers(l_bounds=0, u_bounds=max_idx, n=popsize, endpoint=True)
        population = [indices_to_values(sample, tune_params) for sample in samples]
        population = [repair(individual, searchspace) for individual in population]
    else:
        population = []
        for _ in range(popsize):
            ind = []
            for key in tune_params:
                ind.append(random.choice(tune_params[key]))
            population.append(ind)
    return population


def differential_evolution(searchspace, cost_func, bounds, popsize, maxiter, F, CR, method, constraint_aware, verbose):
    """
    A basic implementation of the Differential Evolution algorithm.

    This function finds the minimum of a given cost function within specified bounds.
    """
    tune_params = cost_func.tuning_options.tune_params
    min_idx = np.zeros(len(tune_params))
    max_idx = [len(v) - 1 for v in tune_params.values()]

    best, mutation, mutation_method, crossover_method = parse_method(method)

    # --- 1. Initialization ---

    # Convert bounds to a numpy array for easier manipulation
    bounds = np.array(bounds)

    # Initialize the population with random individuals within the bounds
    population = generate_population(tune_params, max_idx, popsize, searchspace, constraint_aware)

    # Override with user-specified starting position
    population[0] = cost_func.get_start_pos()

    # Calculate the initial cost for each individual in the population
    population_cost = np.array([cost_func(ind) for ind in population])

    # Keep track of the best solution found so far
    best_idx = np.argmin(population_cost)
    best_solution = population[best_idx]
    best_solution_idx = values_to_indices(best_solution, tune_params)
    best_cost = population_cost[best_idx]

    # --- 2. Main Loop ---

    stabilized = 0

    # Iterate through the specified number of generations
    for generation in range(maxiter):

        # Trial population and vectors are stored as lists
        # not Numpy arrays, to make it easy to check for duplicates
        trial_population = []

        # If for two generations there has been no change, generate a new population
        if stabilized > 2:
            trial_population = list(generate_population(tune_params, max_idx, popsize, searchspace, constraint_aware))

        # Iterate over each individual in the population
        i = 0
        stuck = 0
        while len(trial_population) < popsize:

            # --- a. Mutation ---
            # Select three distinct random individuals (a, b, c) from the population,
            # ensuring they are different from the current individual 'i'.
            idxs = [idx for idx in range(popsize) if idx != i]
            randos = random_draw(idxs, mutation, best)

            if mutation_method == mutate_currenttobest1:
                randos[0] = i

            randos_idx = [values_to_indices(population[rando], tune_params) for rando in randos]

            # Apply mutation strategy
            donor_vector_idx = mutation_method(best_solution_idx, randos_idx, F, min_idx, max_idx, best)
            donor_vector = indices_to_values(donor_vector_idx, tune_params)

            # --- b. Crossover ---
            trial_vector = crossover_method(donor_vector, population[i], CR)

            # Repair if constraint_aware
            if constraint_aware:
                trial_vector = repair(trial_vector, searchspace)

            # Store for selection, if not in trial_population already
            if list(trial_vector) not in trial_population:
                trial_population.append(list(trial_vector))
                i += 1
                stuck = 0
            else:
                stuck += 1

            if stuck >= 20:
                if verbose:
                    print("Differential Evolution got stuck generating new individuals, insert random sample")
                trial_population.append(list(searchspace.get_random_sample(1)[0]))
                i += 1
                stuck = 0


        # --- c. Selection ---

        # Calculate the cost of the new trial vectors
        trial_population_cost = np.array([cost_func(ind) for ind in trial_population])

        # Keep track of whether population changes over time
        no_change = True

        # Iterate over each individual in the trial population
        for i in range(popsize):

            trial_vector = trial_population[i]
            trial_cost = trial_population_cost[i]

            # If the trial vector has a lower or equal cost, it replaces the
            # target vector in the population for the next generation.
            if trial_cost <= population_cost[i]:

                # check if trial_vector is not already in population
                if population.count(trial_vector) == 0:
                    population[i] = trial_vector
                    population_cost[i] = trial_cost
                    no_change = False

                # Update the overall best solution if the new one is better
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_solution = trial_vector
                    best_solution_idx = values_to_indices(best_solution, tune_params)

        # Note if population is stabilizing
        if no_change:
            stabilized += 1

        # Print the progress at the end of the generation
        if verbose:
            print(f"Generation {generation + 1}, Best Cost: {best_cost:.6f}")

    if verbose:
        print(f"Differential Evolution completed fevals={len(cost_func.tuning_options.unique_results)}")

    return {"solution": best_solution, "cost": best_cost}


def round_and_clip(mutant_idx_float, min_idx, max_idx):
    """Helper func to round floating index to nearest integer and clip within bounds."""
    # Round to the nearest integer
    rounded_idx = np.round(mutant_idx_float)

    # Clip the indices to ensure they are within valid index bounds
    clipped_idx = np.clip(rounded_idx, min_idx, max_idx)

    # Convert final mutant vector to integer type
    return clipped_idx.astype(int)


def mutate_currenttobest1(best_idx, randos_idx, F, min_idx, max_idx, best):
    """
    Performs the DE/1 currenttobest1 mutation strategy.

    This function operates on the indices of the parameters, not their actual values.
    The formula v = cur + F * (best - cur + a - b) is applied to the indices, and the result is
    then rounded and clipped to ensure it remains a valid index.
    """
    cur_idx, b_idx, c_idx = randos_idx

    # Apply the DE/currenttobest/1 formula to the indices
    mutant_idx_float = cur_idx + F * (best_idx - cur_idx + b_idx - c_idx)

    return round_and_clip(mutant_idx_float, min_idx, max_idx)


def mutate_randtobest1(best_idx, randos_idx, F, min_idx, max_idx, best):
    """
    Performs the DE/1 randtobest1 mutation strategy.

    This function operates on the indices of the parameters, not their actual values.
    The formula v = a + F * (best - a + b - c) is applied to the indices, and the result is
    then rounded and clipped to ensure it remains a valid index.
    """
    a_idx, b_idx, c_idx = randos_idx

    # Apply the DE/currenttobest/1 formula to the indices
    mutant_idx_float = a_idx + F * (best_idx - a_idx + b_idx - c_idx)

    return round_and_clip(mutant_idx_float, min_idx, max_idx)


def mutate_de_1(best_idx, randos_idx, F, min_idx, max_idx, best):
    """
    Performs the DE/1 mutation strategy.

    This function operates on the indices of the parameters, not their actual values.
    The formula v = a + F * (b - c) is applied to the indices, and the result is
    then rounded and clipped to ensure it remains a valid index.

    """
    if best:
        a_idx = best_idx
        b_idx, c_idx = randos_idx
    else:
        a_idx, b_idx, c_idx = randos_idx

    # Apply the DE/rand/1 formula to the indices
    mutant_idx_float = a_idx + F * (b_idx - c_idx)

    return round_and_clip(mutant_idx_float, min_idx, max_idx)


def mutate_de_2(best_idx, randos_idx, F, min_idx, max_idx, best):
    """
    Performs the DE/2 mutation strategy for a discrete search space.

    This function operates on the indices of the parameters, not their actual values.
    The formula v = a + F1 * (b - c) + F2 * (d - e) is applied to the indices,
    and the result is then rounded and clipped to ensure it remains a valid index.

    """
    if best:
        a_idx = best_idx
        b_idx, c_idx, d_idx, e_idx = randos_idx
    else:
        a_idx, b_idx, c_idx, d_idx, e_idx = randos_idx

    # Apply the DE/2 formula to the indices
    mutant_idx_float = a_idx + F * (b_idx + c_idx - d_idx - e_idx)

    return round_and_clip(mutant_idx_float, min_idx, max_idx)


def binomial_crossover(donor_vector, target, CR):
    """Performs binomial crossover of donor_vector with target given crossover rate CR."""
    # Create the trial vector by mixing parameters from the target and donor vectors
    trial_vector = target.copy()
    dimensions = len(donor_vector)

    # Generate a random array of floats for comparison with the crossover rate CR
    crossover_points = np.random.rand(dimensions) < CR

    # Ensure at least one parameter is taken from the donor vector
    # to prevent the trial vector from being identical to the target vector.
    if not np.any(crossover_points):
        crossover_points[np.random.randint(0, dimensions)] = True

    # Apply crossover
    for i, d in enumerate(donor_vector):
        if crossover_points[i]:
            trial_vector[i] = donor_vector[i]

    return trial_vector


def exponential_crossover(donor_vector, target, CR):
    """
    Performs exponential crossover for a discrete search space.

    This creates a trial vector by taking a contiguous block of parameters
    from the donor vector and the rest from the target vector.
    """
    dimensions = len(target)
    trial_vector = target.copy()

    # 1. Select a random starting point for the crossover block.
    start_point = np.random.randint(0, dimensions)

    # 2. Determine the length of the block to be copied from the mutant.
    # The loop continues as long as random numbers are less than CR.
    # This ensures at least one parameter is always taken from the mutant.
    l = 0
    while np.random.rand() < CR and l < dimensions:
        crossover_point = (start_point + l) % dimensions
        trial_vector[crossover_point] = donor_vector[crossover_point]
        l += 1

    return trial_vector


def repair(trial_vector, searchspace):
    """
    Attempts to repair trial_vector if trial_vector is invalid
    """
    if not searchspace.is_param_config_valid(tuple(trial_vector)):
        # search for valid configurations neighboring trial_vector
        # start from strictly-adjacent to increasingly allowing more neighbors
        for neighbor_method in ["strictly-adjacent", "adjacent", "Hamming"]:
            neighbors = searchspace.get_neighbors_no_cache(tuple(trial_vector), neighbor_method=neighbor_method)

            # if we have found valid neighboring configurations, select one at random
            if len(neighbors) > 0:
                new_trial_vector = list(random.choice(neighbors))
                print(f"Differential evolution resulted in invalid config {trial_vector=}, repaired to {new_trial_vector=}")
                return new_trial_vector

    return trial_vector


mutation = {
    "1": mutate_de_1,
    "2": mutate_de_2,
    "currenttobest": mutate_currenttobest1,
    "randtobest": mutate_randtobest1,
}
crossover = {"bin": binomial_crossover, "exp": exponential_crossover}
