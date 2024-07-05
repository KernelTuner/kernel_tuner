from kernel_tuner.generation.rules.rule import RuleABC
from kernel_tuner.generation.tree.tree import Tree
from kernel_tuner.generation.code.context import Context
from kernel_tuner.generation.token.pragma_token import PRAGMA_TOKEN_TYPE, PRAGMA_KEYWORDS, build_pragma_token
from kernel_tuner.generation.code.line import Line
from kernel_tuner.util import write_file
from kernel_tuner.generation.utils.util import *

class AddNumThreadsAndDistributeRule(RuleABC):

  def __init__(self, tree: Tree, context: Context, initila_params: dict):
    super().__init__(tree, context, initila_params)

  def run(self, debug_file=None):

    parallel_pragmas = filter_pragmas_by_type(self.tree.pragma_tokens, PRAGMA_TOKEN_TYPE.PARALLEL)
    parallel_for_pragmas = filter_pragmas_contains_keyword(parallel_pragmas, PRAGMA_KEYWORDS.FOR)

    old_tokens = []
    new_tokens = []
    new_tune_params = self.generate_param()
    new_meta = dict(map(lambda x: (x[0], x[1]), new_tune_params))
    for parallel_for_pragma in parallel_for_pragmas:
      new_node = build_pragma_token(
        PRAGMA_TOKEN_TYPE.TEAMS,
        [PRAGMA_KEYWORDS.NUM_TEAMS, PRAGMA_KEYWORDS.DISTRIBUTE, PRAGMA_KEYWORDS.PARALLEL, PRAGMA_KEYWORDS.FOR, PRAGMA_KEYWORDS.NUM_THREADS],
        parallel_for_pragma.line.line_number,
        meta = new_meta,
        is_target_used=True,
      )
      old_tokens.append(parallel_for_pragma)
      new_tokens.append(new_node)
    self.context.offer_with_new_token(old_tokens, new_tokens, self.rule_id, new_tune_params)
    
  def generate_param(self) -> PragmaTuneParams:
    param_name = self.context.get_tune_param_unique_name('nteams')
    return self.initial_params + [
      (PRAGMA_KEYWORDS.NUM_TEAMS, param_name, ['1', '2'])
    ]

"""
#pragma omp target parallel num_threads(nthreads)
#pragma omp for schedule(static)


#pragma omp target parallel num_threads(nthreads)
#pragma omp for schedule(static, chunk_size)
"""