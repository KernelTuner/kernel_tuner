"""Wrapper intended for user-defined custom optimization methods"""

from abc import ABC, abstractmethod

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc


class OptAlg(ABC):
    """Base class for user-defined optimization algorithms."""

    def __init__(self):
        self.costfunc_kwargs = {"scaling": True, "snap": True}

    @abstractmethod
    def __call__(self, func: CostFunc, searchspace: Searchspace) -> tuple[tuple, float]:
        """Optimize the black box function `func` within the given `searchspace`.

        Args:
            func (CostFunc): Cost function to be optimized. Has a property `budget_spent_fraction` that indicates how much of the budget has been spent.
            searchspace (Searchspace): Search space containing the parameters to be optimized.

        Returns:
            tuple[tuple, float]: tuple of the best parameters and the corresponding cost value
        """
        pass


class OptAlgWrapper:
    """Wrapper class for user-defined optimization algorithms"""

    def __init__(self, optimizer: OptAlg):
        self.optimizer: OptAlg = optimizer

    def tune(self, searchspace: Searchspace, runner, tuning_options):
        cost_func = CostFunc(searchspace, tuning_options, runner, **self.optimizer.costfunc_kwargs)

        if self.optimizer.costfunc_kwargs.get('scaling', True):
            # Initialize costfunc for scaling
            cost_func.get_bounds_x0_eps()

        try:
            self.optimizer(cost_func, searchspace)
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)

        return cost_func.results
