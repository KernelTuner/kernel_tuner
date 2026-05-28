"""
Hybrid VND with surrogate modeling, adaptive neighborhoods, and annealing.

Algorithm generated as part of the paper "Automated Algorithm Design For Auto-Tuning Optimizers".
"""

import random
import math
import collections
import heapq

from kernel_tuner.util import StopCriterionReached
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc


_options = dict(
    budget=("maximum number of evaluations", 5000),
    k=("k for k-NN surrogate", 5),
    cand_pool=("candidate pool size", 8),
    restart_iter=("iterations before restart", 100),
    tabu_size=("tabu list size", 300),
    elite_size=("elite set size", 5),
    temp0=("initial temperature", 1.0),
    cooling=("cooling rate", 0.995),
    constraint_aware=("constraint-aware optimization (True/False)", True),
)


def tune(searchspace, runner, tuning_options):

    options = tuning_options.strategy_options
    if "x0" in options:
        raise ValueError("Starting point (x0) is not supported for HybridVNDX strategy.")
    budget, k, cand_pool, restart_iter, tabu_size, elite_size, temp0, cooling, constraint_aware = common.get_options(options, _options)

    cost_func = CostFunc(searchspace, tuning_options, runner)

    alg = HybridVNDX(
        searchspace,
        cost_func,
        budget,
        k,
        cand_pool,
        restart_iter,
        tabu_size,
        elite_size,
        temp0,
        cooling,
        constraint_aware,
        tuning_options.verbose,
    )

    try:
        alg.run()
    except StopCriterionReached as e:
        if tuning_options.verbose:
            print(e)

    return cost_func.results


tune.__doc__ = common.get_strategy_docstring("Hybrid VNDX", _options)


class HybridVNDX:

    def __init__(self, searchspace, cost_func, budget, k, cand_pool,
                 restart_iter, tabu_size, elite_size, temp0, cooling,
                 constraint_aware, verbose):

        self.searchspace = searchspace
        self.cost_func = cost_func
        self.budget = budget
        self.k = k
        self.cand_pool = cand_pool
        self.restart_iter = restart_iter
        self.tabu_size = tabu_size
        self.elite_size = elite_size
        self.temp0 = temp0
        self.cooling = cooling
        self.constraint_aware = constraint_aware
        self.verbose = verbose

        self.neighbor_methods = ["strictly-adjacent", "adjacent", "Hamming"]

    def sample_valid(self):
        while True:
            x = list(self.searchspace.get_random_sample(1)[0])
            if not self.constraint_aware or self.searchspace.is_param_config_valid(tuple(x)):
                return x

    def repair(self, x):
        if not self.constraint_aware or self.searchspace.is_param_config_valid(tuple(x)):
            return x
        for _ in range(5):
            y = list(self.searchspace.get_random_sample(1)[0])
            if self.searchspace.is_param_config_valid(tuple(y)):
                return y
        return x

    def knn_predict(self, tpl, history):
        dists = [(sum(a != b for a, b in zip(tpl, xh)), fh) for xh, fh in history]
        dists.sort(key=lambda z: z[0])
        top = dists[:self.k]
        return sum(f for _, f in top) / len(top)

    def pick_nm(self, nm_weight):
        total = sum(nm_weight.values())
        r = random.random() * total
        cum = 0
        for nm, w in nm_weight.items():
            cum += w
            if r <= cum:
                return nm
        return self.neighbor_methods[-1]

    def evaluate(self, dna):
        return self.cost_func(dna, check_restrictions=not self.constraint_aware)

    def run(self):

        curr = self.sample_valid()
        curr_f = self.evaluate(curr)

        best = list(curr)
        best_f = curr_f

        history = [(tuple(curr), curr_f)]

        tabu = collections.deque(maxlen=self.tabu_size)
        tabu.append(tuple(curr))

        elite = [(curr_f, tuple(curr))]

        nm_weight = {nm: 1.0 for nm in self.neighbor_methods}

        no_improve = 0
        temp = self.temp0
        num_evals = 1

        while num_evals < self.budget:

            nm = self.pick_nm(nm_weight)

            # generate candidate pool
            pool = []

            nbrs = self.searchspace.get_neighbors(tuple(curr), neighbor_method=nm) or []
            if nbrs:
                nsel = min(len(nbrs), self.cand_pool // 2)
                pool += random.sample(nbrs, nsel)

            # elite crossover
            if len(elite) >= 2:
                (_, x1), (_, x2) = random.sample(elite, 2)
                child = [random.choice((a, b)) for a, b in zip(x1, x2)]
                pool.append(tuple(child))

            # random fill
            while len(pool) < self.cand_pool:
                pool.append(tuple(self.searchspace.get_random_sample(1)[0]))

            # repair + deduplicate
            seen = set()
            clean = []
            for c in pool:
                rc = tuple(self.repair(list(c)))
                if rc not in seen:
                    seen.add(rc)
                    clean.append(rc)

            # surrogate scoring
            scored = []
            for c in clean:
                s = self.knn_predict(c, history)
                if c in tabu:
                    s += abs(s) * 0.1 + 1e3
                scored.append((s, c))

            _, cand_tpl = min(scored, key=lambda x: x[0])
            cand = list(cand_tpl)

            try:
                f_c = self.evaluate(cand)
                num_evals += 1
            except StopCriterionReached:
                raise

            history.append((tuple(cand), f_c))

            # update elite
            heapq.heappush(elite, (f_c, tuple(cand)))
            if len(elite) > self.elite_size:
                heapq.heappop(elite)

            # acceptance (SA-style)
            delta = f_c - curr_f
            accept = (delta < 0) or (random.random() < math.exp(-delta / max(temp, 1e-8)))

            if accept:
                tabu.append(tuple(cand))
                curr, curr_f = list(cand), f_c

                nm_weight[nm] *= 1.1
                no_improve = 0

                if f_c < best_f:
                    best, best_f = list(cand), f_c
            else:
                nm_weight[nm] *= 0.9
                no_improve += 1

            # normalize weights
            if sum(nm_weight.values()) > 1e6:
                for k in nm_weight:
                    nm_weight[k] *= 1e-6

            temp *= self.cooling

            # restart
            if no_improve >= self.restart_iter:
                curr = self.sample_valid()
                curr_f = self.evaluate(curr)
                num_evals += 1

                history.append((tuple(curr), curr_f))

                tabu.clear()
                tabu.append(tuple(curr))

                no_improve = 0
                temp = self.temp0

            if self.verbose and num_evals % 50 == 0:
                print(f"Evaluations: {num_evals}, best: {best_f}")