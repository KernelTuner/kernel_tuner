from kernel_tuner.generation.rules.rule import RuleABC
from kernel_tuner.generation.tree.tree import Tree
from kernel_tuner.generation.code.context import Context
from kernel_tuner.generation.token.pragma_token import PRAGMA_TOKEN_TYPE, PRAGMA_KEYWORDS, build_pragma_token
from kernel_tuner.generation.code.line import Line
from kernel_tuner.util import write_file
from kernel_tuner.generation.utils.util import *

# Only static schedule kind is supported for GPU
class AddChunkSizeToScheduleRule(RuleABC):

  def __init__(self, tree: Tree, context: Context, initila_params: dict):
    super().__init__(tree, context, initila_params)

  def run(self, debug_file=None):
    pragma_for = filter_pragmas_contains_keyword(self.tree.pragma_tokens, [PRAGMA_KEYWORDS.SCHEDULE])
    new_params = self.generate_param()
    for pragma_for_child in pragma_for:
      if PRAGMA_KEYWORDS.SCHEDULE in pragma_for_child.meta:
        schedule = pragma_for_child.meta[PRAGMA_KEYWORDS.SCHEDULE].strip()
        if schedule == 'static':
          pragma_for_child.meta[PRAGMA_KEYWORDS.SCHEDULE] = 'static, chunk_size'
          pragma_for_child.modify_keywords([], pragma_for_child.meta)
          self.context.offer_with_new_token([pragma_for_child], [pragma_for_child], self.rule_id, new_params)


  def generate_param(self) -> PragmaTuneParams:
    param_name = self.context.get_tune_param_unique_name('chunk_size')
    return self.initial_params + [
      (PRAGMA_KEYWORDS.SCHEDULE, param_name, ['16', '32'])
    ]

"""

#pragma omp target parallel for num_threads(nthreads)

#pragma omp target parallel num_threads(16)
#pragma omp for schedule(dynamic)
"""