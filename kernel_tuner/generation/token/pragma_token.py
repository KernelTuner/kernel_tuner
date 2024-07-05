from __future__ import annotations
import copy
from kernel_tuner.generation.token.token import *
from kernel_tuner.generation.code.line import Line
from kernel_tuner.generation.code.code import CodeBlock, Code
import numpy as np
import re


class PragmaToken(Token):

  def __init__(self, line: Line, level) -> None:
    self.initial_line = copy.deepcopy(line)
    super().__init__(line, CodeBlock([line]), TOKEN_TYPE.PRAGMA)
    self.content = CodeBlock([line])
    self.line.replace('#pragma omp')
    if self.line.startswith('target'):
      self.is_target_used = True
      self.line.replace('target')
    else:
      self.is_target_used = False
    
    self.pragma_type = self.__detect__pragma_type()
    (self.keywords, self.meta) = self.__detect_keywords()
    self.pragma_children = []

    if self.pragma_type.is_data():
      self.level = 1
    else:
      self.level = level
  
  def append_child(self, child):
    super().append_child(child)
    if type(child) is not PragmaToken:
      new_content = [self.initial_line] + child.content.lines
      self.content = CodeBlock(new_content)
    else:
      self.pragma_children.append(child)

  def remove_child(self, child):
        super().remove_child(child)

  def modify_keywords(
      self, 
      new_keywords: list[PRAGMA_KEYWORDS],
      meta: dict[PRAGMA_KEYWORDS, str] = {},
      replace_keywords: list[PRAGMA_KEYWORDS] = []
    ):
    new_kw = list(filter(lambda x: x not in replace_keywords, self.keywords))
    new_kw += [x for x in new_keywords if x not in new_kw]
    self.keywords = new_kw
    self.meta.update(meta)
    self.__rebuild()

  def find_first_pragma(self, type: PRAGMA_TOKEN_TYPE) -> PragmaToken|None:
    queue:list[PragmaToken] = [self]
    results = []
    self.__bfs_pragma(type, queue, results)
    if len(results) > 0:
      return results.pop(0)
    return None
  
  def find_all_pragma(self, type: PRAGMA_TOKEN_TYPE) -> list[PragmaToken]:
    queue:list[PragmaToken] = [self]
    results = []
    self.__bfs_pragma(type, queue, results)
    return results

  def __bfs_pragma(self, type: PRAGMA_TOKEN_TYPE, queue: list[PragmaToken], results: list[PragmaToken]):
    if len(queue) == 0:
      return None
    cur_tkn = queue.pop(0)
    for ch in cur_tkn.pragma_children:
      if ch.pragma_type == type:
          results.append(ch)
      queue.append(ch)
    self.__bfs_pragma(type, queue, results)

  def __detect__pragma_type(self) -> PRAGMA_TOKEN_TYPE:
    if self.line.startswith('enter data'):
      return PRAGMA_TOKEN_TYPE.DATA_ENTER
    elif self.line.startswith('exit data'):
      return PRAGMA_TOKEN_TYPE.DATA_EXIT
    for ptk in PRAGMA_TOKEN_TYPE:
      if self.line.startswith(ptk.name.lower()):
        return ptk
    return PRAGMA_TOKEN_TYPE.UNKNOWN
  
  def __detect_keywords(self) -> tuple[list[PRAGMA_KEYWORDS], dict[PRAGMA_KEYWORDS, str]]:
    keywords_result = []
    keywords_map = {}
    pattern_with_parentheses = re.compile(r'({})\(.*\)'.format('|'.join(PRAGMA_KEYWORDS_VALUES)))
    pattern_exact = re.compile(r'({})'.format('|'.join(PRAGMA_KEYWORDS_VALUES)))
    line = self.line
    for word in line.content.split():
      is_pattern_with_parentheses = pattern_with_parentheses.match(word)
      is_pattern_exact = pattern_exact.match(word)
      if is_pattern_with_parentheses:
        for key_word in PRAGMA_KEYWORDS_VALUES:
          if re.match(r'{}\(.*\)'.format(key_word), word):
            if '(' in word: 
              param = re.split(r'[()]', word)
              keywords_map[PRAGMA_KEYWORDS(param[0].strip())] = param[1].strip()
            keywords_result.append(PRAGMA_KEYWORDS(key_word))
      elif is_pattern_exact:
        keywords_result.append(PRAGMA_KEYWORDS(is_pattern_exact.group(0)))
    
    return (keywords_result, keywords_map)

  def __rebuild(self):
    kws = ''
    for kw in self.keywords:
      kws += f"{kw.name.lower()} "
      if kw in self.meta:
        kws += f"({self.meta[kw]}) "
    target = " target " if self.is_target_used else ""
    self.initial_line = Line(f"#pragma omp{target} {kws}", self.initial_line.line_number)

  def print(self, debug=False) -> str:
    result = f"id: {self.id}\n"
    result += f"type: {self.type}\n"
    result += f"level: {self.level}\n"
    result += f"pragma_type: {self.pragma_type}\n"
    result += f"keywords: {list(map(lambda x: x.name, self.keywords))}\n"
    result += f"META: {self.meta}\n"
    result += f"line_start: {self.initial_line.content}\n"
    result += f"children: {list(map(lambda x: x.id, self.children))}\n"
    if debug:
      result += f"content: \n {self.print_content()}\n"
    return result


def build_pragma_token(
    type: PRAGMA_TOKEN_TYPE,
    keywords: list[PRAGMA_KEYWORDS],
    line_number: int,
    meta: dict[PRAGMA_KEYWORDS, str] = {},
    is_target_used: bool = True,
    level: int|None = None
  ) -> PragmaToken:

  target = "target" if is_target_used else ""

  pragma_type_str = f"{type.name.lower()}"
  
  for keyword in keywords:
    pragma_type_str += f" {keyword.name.lower()}"
    if keyword in meta:
      pragma_type_str += f" ({meta[keyword]})"

  line = f"#pragma omp {target} {pragma_type_str}"
  return PragmaToken(Line(line, line_number), level if level else 0)