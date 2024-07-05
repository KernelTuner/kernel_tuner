from abc import ABC, abstractmethod
from kernel_tuner.generation.tree.tree import Tree
from kernel_tuner.generation.code.context import Context
from kernel_tuner.generation.utils.util import PragmaTuneParams
import random


class RuleABC(ABC):

  def __init__(self, tree: Tree, context: Context, initila_params: PragmaTuneParams):
    self.tree = tree
    self.context = context
    self.initial_params = initila_params
    self.rule_id = random.randint(1, 100)
    
  @abstractmethod
  def run(self, debug_file=None):
    pass

  @abstractmethod
  def generate_param(self):
    pass