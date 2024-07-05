from kernel_tuner.generation.code.code import Code
from kernel_tuner.generation.code.context import Context
from kernel_tuner.generation.tree.tree import TreeBuilder, Tree
from kernel_tuner.generation.rules.add_num_threads_and_distribute_rule import AddNumThreadsAndDistributeRule
from kernel_tuner.generation.rules.add_chunk_size_to_schedule_rule import AddChunkSizeToScheduleRule
from kernel_tuner.generation.rules.add_schedule_rule import AddStaticScheduleRule
from kernel_tuner.generation.token.pragma_token import PragmaToken, PRAGMA_KEYWORDS
from kernel_tuner.util import write_file
from kernel_tuner.generation.utils.util import PragmaTuneParams, convertPragmaTuneToDict

def generate_kernel_sources(initial_code_str: str, initilea_tune_params: dict, debug_file=None):
  print(initilea_tune_params)
  code = Code(initial_code_str.split('\n'))
  if debug_file:
    write_file(debug_file, '='*10 + 'CODE' + '='*10 + '\n' + code.to_text() + '\n\n', "a")
  tree_builder = TreeBuilder(code)
  tree = tree_builder.build_tree()
  if debug_file:
    write_file(debug_file, '='*10 + 'TREE' + '='*10 + '\n\n', "a")
    tree.dfs_print(debug_file_name=debug_file)
  
  result_tune_param = []
  tree.dfs(convert_pragma_keyword_to_initiail_tune_params, initilea_tune_params, result_tune_param)
  context = Context(code, result_tune_param)
  write_file(debug_file, '='*10 + 'RULES' + '='*10 + '\n\n', "a")

  #s_rule = AddNumThreadsAndDistributeRule(tree, store, result_tune_param)
  # s_rule = AddChunkSizeToScheduleRule(tree, store, result_tune_param)
  s_rule = AddStaticScheduleRule(tree, context, result_tune_param)
  s_rule.run(debug_file)
  result = [(code, result_tune_param)] + [context.get(s_rule.rule_id)]
  return post_process(result)


def convert_pragma_keyword_to_initiail_tune_params(
    node: PragmaToken,    
    initial_tune_params: dict[str, str],
    result_tune_param: PragmaTuneParams
  ):
  for (pragma_keyword, param_name) in node.meta.items():
    if param_name in initial_tune_params:
      result_tune_param.append((pragma_keyword, param_name, initial_tune_params[param_name]))

def post_process(result: list[tuple[Code, PragmaTuneParams]]):
  return list(map(lambda x: (x[0], convertPragmaTuneToDict(x[1])), result))