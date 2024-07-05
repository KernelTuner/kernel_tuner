from kernel_tuner.generation.rules.rule import RuleABC
from kernel_tuner.generation.tree.tree import Tree
from kernel_tuner.generation.code.context import Context
from kernel_tuner.generation.token.pragma_token import PRAGMA_TOKEN_TYPE, PRAGMA_KEYWORDS, build_pragma_token
from kernel_tuner.generation.code.line import Line
from kernel_tuner.util import write_file
from kernel_tuner.generation.utils.util import *

# Only static schedule kind is supported for GPU
class AddStaticScheduleRule(RuleABC):

  def __init__(self, tree: Tree, context: Context, initila_params: dict):
    super().__init__(tree, context, initila_params)

  def run(self, debug_file=None):
    new_params = self.generate_param()
    pragma_for = filter_pragmas_contains_keyword(self.tree.pragma_tokens, [PRAGMA_KEYWORDS.FOR], [PRAGMA_KEYWORDS.SCHEDULE])
    for pragma_for_child in pragma_for:
      # uppper node -> pragma without for
      pragma_for_child.modify_keywords([], replace_keywords=[PRAGMA_KEYWORDS.FOR])

      new_node = build_pragma_token(
        PRAGMA_TOKEN_TYPE.FOR,
        [PRAGMA_KEYWORDS.SCHEDULE],
        pragma_for_child.initial_line.line_number,
        {PRAGMA_KEYWORDS.SCHEDULE: 'scedule_type'},
        is_target_used=False
      )

      for ch in pragma_for_child.children:
        new_node.append_child(ch)
        pragma_for_child.remove_child(ch)
      pragma_for_child.append_child(new_node)

      self.context.offer_with_new_token(
        [pragma_for_child], [new_node], self.rule_id, new_params
      )

      self.context.offer_with_add_pragma_above(
        new_node, pragma_for_child, self.rule_id, new_params
      )


  def generate_param(self) -> PragmaTuneParams:
    param_name = self.context.get_tune_param_unique_name('scedule_type')
    return self.initial_params + [
      (PRAGMA_KEYWORDS.SCHEDULE, param_name, ['static'])
    ]

"""

#pragma omp target parallel for num_threads(nthreads)

#pragma omp target parallel num_threads(16)
#pragma omp for schedule(dynamic)
"""