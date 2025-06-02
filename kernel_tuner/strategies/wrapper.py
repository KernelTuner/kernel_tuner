"""Wrapper intended for user-defined custom optimization methods"""

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc


class OptAlgWrapper:
    """Wrapper class for user-defined optimization algorithms"""

    def __init__(self, optimizer, scaling=True):
        self.optimizer = optimizer
        self.scaling = scaling


    def tune(self, searchspace: Searchspace, runner, tuning_options):
        cost_func = CostFunc(searchspace, tuning_options, runner, scaling=self.scaling)

        if self.scaling:
            # Initialize costfunc for scaling
            cost_func.get_bounds_x0_eps()

        try:
            self.optimizer(cost_func, searchspace)
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)

        return cost_func.results
