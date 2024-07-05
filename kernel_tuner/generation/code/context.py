import copy
from kernel_tuner.generation.code.code import Code
from kernel_tuner.generation.code.line import Line
from kernel_tuner.generation.token.pragma_token import PragmaToken, PRAGMA_KEYWORDS
from kernel_tuner.generation.utils.util import PragmaTuneParams

class Context:

  def __init__(self, initial_code: Code, initial_tune_params: PragmaTuneParams):
    self.initial_code = initial_code
    self.tune_param_names = set(map(lambda x: x[1], initial_tune_params))
    self.propositions: dict[str, tuple[Code, PragmaTuneParams]] = {}

  # def offer_with_new_line(self, old_line: Line, new_line:Line, rule_id: str):
  #   proposition_code = copy.deepcopy(self.initial_code)
  #   for idx, line in enumerate(proposition_code.initial_lines):
  #     if line.line_number == old_line.line_number:
  #       proposition_code.initial_lines[idx] = new_line
  #       break
  #   self.propositions.append(proposition_code)

  def __append(
      self, 
      rule_id: str,
      proposition_code: Code,
      pragma_tune_params: PragmaTuneParams
  ):
    self.propositions[rule_id] = (proposition_code, pragma_tune_params)
    self.tune_param_names.add(set(map(lambda x: x[1], pragma_tune_params)))

  def offer_with_new_lines(
      self, 
      old_lines: list[Line], 
      new_lines: list[Line], 
      rule_id: str,
      pragma_tune_params: PragmaTuneParams
    ):
    if len(old_lines) != len(new_lines):
      return
    proposition_code = copy.deepcopy(self.initial_code)
    for idx, line in enumerate(proposition_code.initial_lines):
      for old_idx, old in enumerate(old_lines):
        if line.line_number == old.line_number:
          proposition_code.initial_lines[idx] = new_lines[old_idx] 
          continue
    self.__append(rule_id, proposition_code, pragma_tune_params)


  def offer_with_new_token(
      self, 
      old_tokens: list[PragmaToken], 
      new_tokens: list[PragmaToken], 
      rule_id: str,
      pragma_tune_params: PragmaTuneParams
    ):
    self.offer_with_new_lines(
      list(map(lambda x: x.initial_line, old_tokens)),
      list(map(lambda x: x.initial_line, new_tokens)),
      rule_id,
      pragma_tune_params
    )


  def offer_with_new_token_content(
      self, 
      old_token: PragmaToken,
      new_token: PragmaToken,
      rule_id: str,
      pragma_tune_params: PragmaTuneParams
    ):
    if rule_id in self.propositions:
      proposition_code = self.propositions[rule_id][0]
    else:
      proposition_code = copy.deepcopy(self.initial_code)
    for idx, line in enumerate(proposition_code.initial_lines):
      if line.line_number == old_token.initial_line.line_number:
        proposition_code.initial_lines = proposition_code.initial_lines[:idx] + new_token.content.lines + proposition_code.initial_lines[idx + len(old_token.content.lines):]
        break
    self.__append(rule_id, proposition_code, pragma_tune_params)

  def offer_with_add_pragma_above(
      self,
      old_token: PragmaToken,
      new_token: PragmaToken,
      rule_id: str,
      pragma_tune_params: PragmaTuneParams
  ):
    if rule_id in self.propositions:
      proposition_code = self.propositions[rule_id][0]
    else:
      proposition_code = copy.deepcopy(self.initial_code)
    for idx, line in enumerate(proposition_code.initial_lines):
      if line.line_number == old_token.initial_line.line_number:
        proposition_code.initial_lines.insert(idx, new_token.initial_line)
        break
    self.__append(rule_id, proposition_code, pragma_tune_params)


  def offer_with_line_token(self, old_line: Line, new_token: PragmaToken):
    proposition_code = copy.deepcopy(self.initial_code)
    for idx, line in enumerate(proposition_code.initial_lines):
      if line.line_number == old_line.line_number:
        proposition_code.initial_lines = proposition_code.initial_lines[:idx-1] + new_token.content.lines + proposition_code.initial_lines[idx:]
        break
    self.propositions.append(proposition_code)

  # TODO make it more beautiful
  def get_tune_param_unique_name(self, prefix: str) -> str:
    while prefix in self.tune_param_names:
      prefix += '_A'
    return prefix
    

  def get(self, rule_id) -> tuple[Code, PragmaTuneParams]:
    if rule_id in self.propositions:
      return self.propositions[rule_id]
    return None

  def print_propositions(self):
    for proposition in self.propositions:
      proposition.print()