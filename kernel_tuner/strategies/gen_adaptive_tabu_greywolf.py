"""
Adaptive Tabu-Guided Grey Wolf Optimization. 

Algorithm generated as part of the paper "Automated Algorithm Design For Auto-Tuning Optimizers". 
"""

import random
import math
from collections import deque

from kernel_tuner.util import StopCriterionReached
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc


_options = dict(
    budget=("maximum number of evaluations", 5000),
    pack_size=("number of wolves", 8),
    tabu_factor=("tabu size multiplier", 3),
    shake_rate=("base shaking probability", 0.2),
    jump_rate=("random jump probability", 0.15),
    stagn_limit=("stagnation limit before restart", 80),
    restart_ratio=("fraction of pack to restart", 0.3),
    t0=("initial temperature", 1.0),
    t_decay=("temperature decay rate", 5.0),
    t_min=("minimum temperature", 1e-4),
    constraint_aware=("constraint-aware optimization (True/False)", True),
)


def tune(searchspace, runner, tuning_options):

    options = tuning_options.strategy_options
    if "x0" in options:
        raise ValueError("Starting point (x0) is not supported for AdaptiveTabuGreyWolf strategy.")
    (budget, pack_size, tabu_factor, shake_rate, jump_rate,
     stagn_limit, restart_ratio, t0, t_decay, t_min, constraint_aware) = \
        common.get_options(options, _options)

    cost_func = CostFunc(searchspace, tuning_options, runner)

    alg = AdaptiveTabuGreyWolf(
        searchspace, cost_func,
        budget, pack_size, tabu_factor,
        shake_rate, jump_rate,
        stagn_limit, restart_ratio,
        t0, t_decay, t_min,
        constraint_aware,
        tuning_options.verbose,
    )

    try:
        alg.run()
    except StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("Adaptive Tabu Grey Wolf", _options)


class AdaptiveTabuGreyWolf:

    def __init__(self, searchspace, cost_func,
                 budget, pack_size, tabu_factor,
                 shake_rate, jump_rate,
                 stagn_limit, restart_ratio,
                 t0, t_decay, t_min,
                 constraint_aware, verbose):

        self.searchspace = searchspace
        self.cost_func = cost_func
        self.budget = budget
        self.pack_size = pack_size
        self.tabu = deque(maxlen=pack_size * tabu_factor)
        self.shake_rate = shake_rate
        self.jump_rate = jump_rate
        self.stagn_limit = stagn_limit
        self.restart_ratio = restart_ratio
        self.t0 = t0
        self.t_decay = t_decay
        self.t_min = t_min
        self.constraint_aware = constraint_aware
        self.verbose = verbose

    def evaluate(self, dna):
        return self.cost_func(dna, check_restrictions=not self.constraint_aware)

    def sample_valid(self):
        while True:
            x = list(self.searchspace.get_random_sample(1)[0])
            if not self.constraint_aware or self.searchspace.is_param_config_valid(tuple(x)):
                return x

    def repair(self, sol):
        if not self.constraint_aware or self.searchspace.is_param_config_valid(tuple(sol)):
            return sol

        # try neighbors
        for m in ("adjacent", "Hamming", "strictly-adjacent"):
            for nb in self.searchspace.get_neighbors(tuple(sol), neighbor_method=m):
                if self.searchspace.is_param_config_valid(nb):
                    return list(nb)

        return self.sample_valid()

    def run(self):

        # initialize pack
        pack = []
        num_evals = 0

        for cfg in self.searchspace.get_random_sample(self.pack_size):
            sol = list(cfg)

            try:
                val = self.evaluate(sol)
                num_evals += 1
            except StopCriterionReached:
                raise

            pack.append((sol, val))
            self.tabu.append(tuple(sol))

        pack.sort(key=lambda x: x[1])

        best_sol, best_val = pack[0]
        stagn = 0
        iteration = 0

        while num_evals < self.budget:

            iteration += 1
            frac = num_evals / self.budget

            # temperature schedule
            T = max(self.t_min, self.t0 * math.exp(-self.t_decay * frac))

            # reheating
            if stagn and stagn % max(1, (self.stagn_limit // 2)) == 0:
                T += self.t0 * 0.2

            # adaptive shaking
            shake_p = min(0.5, self.shake_rate * (1 + stagn / self.stagn_limit))

            pack.sort(key=lambda x: x[1])
            alpha, beta, delta = pack[0][0], pack[1][0], pack[2][0]

            new_pack = []

            for sol, sol_val in pack:

                # leaders survive
                if sol in (alpha, beta, delta):
                    new_pack.append((sol, sol_val))
                    continue

                D = len(sol)

                # recombination
                child = [
                    random.choice((alpha[i], beta[i], delta[i], sol[i]))
                    for i in range(D)
                ]

                # shaking
                if random.random() < shake_p:
                    if random.random() < self.jump_rate:
                        idx = random.randrange(D)
                        rnd = random.choice(self.searchspace.get_random_sample(1))
                        child[idx] = rnd[idx]
                    else:
                        method = "adjacent" if frac < 0.5 else "strictly-adjacent"
                        nbrs = list(self.searchspace.get_neighbors(tuple(child), neighbor_method=method))
                        if nbrs:
                            child = list(random.choice(nbrs))

                # repair
                child = self.repair(child)
                tchild = tuple(child)

                # tabu handling
                if tchild in self.tabu:
                    nbrs = list(self.searchspace.get_neighbors(tchild, neighbor_method="Hamming"))
                    if nbrs:
                        child = list(random.choice(nbrs))

                try:
                    fch = self.evaluate(child)
                    num_evals += 1
                except StopCriterionReached:
                    raise

                self.tabu.append(tuple(child))

                # SA acceptance
                dE = fch - sol_val
                if dE <= 0 or random.random() < math.exp(-dE / T):
                    new_pack.append((child, fch))
                else:
                    new_pack.append((sol, sol_val))

                if num_evals >= self.budget:
                    break

            pack = new_pack
            pack.sort(key=lambda x: x[1])

            # update best
            if pack[0][1] < best_val:
                best_sol, best_val = pack[0]
                stagn = 0
            else:
                stagn += 1

            # restart
            if stagn >= self.stagn_limit:
                nr = int(math.ceil(self.pack_size * self.restart_ratio))

                for i in range(self.pack_size - nr, self.pack_size):
                    sol = self.sample_valid()

                    try:
                        val = self.evaluate(sol)
                        num_evals += 1
                    except StopCriterionReached:
                        raise

                    pack[i] = (sol, val)
                    self.tabu.append(tuple(sol))

                pack.sort(key=lambda x: x[1])
                best_sol, best_val = pack[0]
                stagn = 0

            if self.verbose and num_evals % 50 == 0:
                print(f"Evaluations: {num_evals}, best: {best_val}")