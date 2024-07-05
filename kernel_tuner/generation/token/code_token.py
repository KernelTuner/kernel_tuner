from __future__ import annotations
from kernel_tuner.generation.token.token import *
from kernel_tuner.generation.code.code import Code, CodeBlock
import re


class CodeToken(Token):

  def __init__(self, line: Line, content: CodeBlock, type: TOKEN_TYPE) -> None:
    super().__init__(line, content, type)
    self.__detect_children()

  
  def __detect_children(self):
    idx = 1
    rest_possible_types = [
      CodeToken.build_function_code_token,
      CodeToken.build_variable_declaration,
      CodeToken.build_variable_assignment,
      CodeToken.build_variable_reassignment
    ]

    while idx < self.content.size():

      code_token = None
      next_line = self.content.get(idx)
      if not next_line:
        return

      if next_line.startswith('{'):

        if self.type == TOKEN_TYPE.FOR or self.type == TOKEN_TYPE.IF:
          idx+=1
          continue

        code_token = CodeToken.build_block_code_token(next_line, CodeBlock(self.content.lines[idx:]))

      elif next_line.startswith('for'):
        code_token = CodeToken.build_for_code_token(next_line, CodeBlock(self.content.lines[idx:]))

      elif next_line.startswith('}'):
        idx+=1
        continue

      elif next_line.startswith('if'):
        code_token = CodeToken.build_if_code_token(next_line, CodeBlock(self.content.lines[idx:]))

      else:
        for f in rest_possible_types:
          code_token = f(next_line, CodeBlock(self.content.lines[idx:]))
          if code_token:
            break

      if code_token:
        self.append_child(code_token)
        idx+=len(code_token.content.lines)
      else:
        return

    
  @staticmethod
  def build_code_token(line: Line, initial_code_block: CodeBlock) -> CodeToken | None:
    if line.startswith('{'):
      return CodeToken.build_block_code_token(line, initial_code_block)
    elif line.startswith('for'):
      return CodeToken.build_for_code_token(line, initial_code_block)
    elif line.startswith('if'):
      return CodeToken.build_if_code_token(line, initial_code_block)
    else:
      return CodeToken.build_function_code_token(line, initial_code_block)

  @staticmethod
  def build_block_code_token(statrt_line: Line, initial_code: CodeBlock) -> CodeToken|None:
    node_content = []
    open_braces_count = 0
    idx = 0
    line = statrt_line
    while(True):          
      node_content.append(line)
      if line.is_open_brace():
        open_braces_count+=1
      elif line.is_close_brace():
        open_braces_count-=1
      idx+=1          
      if open_braces_count == 0:
        break
      line = initial_code.get(idx)
      if not line:
        return None
    return CodeToken(statrt_line, CodeBlock(node_content), TOKEN_TYPE.BLOCK)
  
  
  @staticmethod
  def build_for_code_token(start_line: Line, initial_code: CodeBlock) -> CodeToken|None:
    node_content = []
    open_braces_count = 0
    idx = 0
    line = start_line
    while True:
      node_content.append(line)
      if line.is_open_brace():
        open_braces_count+=1
      if len(node_content) == 1 and not line.is_open_brace():
        idx+=1
        next_line = initial_code.get(idx)
        if not next_line:
          return None
        if next_line.is_open_brace():
          open_braces_count+=1
          node_content.append(next_line)
        else:
          return CodeToken.build_one_line_for_code_token(start_line, next_line)
      elif line.is_close_brace():
        open_braces_count-=1
      idx+=1
      if open_braces_count == 0:
        break
      line = initial_code.get(idx)
      if not line:
        print("TODO NOT LINE")
        break
    return CodeToken(start_line, CodeBlock(node_content), TOKEN_TYPE.FOR)

  @staticmethod
  def build_one_line_for_code_token(current_line: Line, next_line: Line) -> CodeToken|None:
    open_parenthesis_count = 1
    start_index = current_line.find('(')
    last_paranthesis_idx = start_index    
    if not start_index:
      return None
    for idx, char in enumerate(current_line.content[start_index+1:]):
      if open_parenthesis_count == 0:
        break
      if char == '(':
        open_parenthesis_count+=1
      elif char == ')':
        open_parenthesis_count-=1
        last_paranthesis_idx=idx

    if (last_paranthesis_idx + start_index + 1) == current_line.len() - 1:
      return CodeToken(current_line, CodeBlock([current_line, next_line]), TOKEN_TYPE.FOR)
    return CodeToken(current_line, CodeBlock([current_line]), TOKEN_TYPE.FOR)

  
  @staticmethod
  def build_function_code_token(start_line: Line, initial_code: CodeBlock):
    idx = 0
    line = start_line
    node_content = []
    function_call_pattern = r'^\b[a-zA-Z_][a-zA-Z0-9_]*\s*\([^;{}]*\)\s*;?\s*$'
    function_start_call_patter = r'^\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*$'
    while True:
      node_content.append(line)
      content_as_string = '\n'.join(list(map(lambda x: x.content, node_content)))
      match = re.search(function_call_pattern, content_as_string, re.DOTALL | re.MULTILINE)
      if not match:
        if len(node_content) == 1:
          match = re.search(function_start_call_patter, line.content)
          if not match:
            return None
        idx+=1
        line = initial_code.get(idx)
        if not line:
          print("TODO: NOT LINE!")
          return None
        continue
      # check only one function is called!
      # if idx_line != line.len()-1:
      #   return None
      break
    return CodeToken(start_line, CodeBlock(node_content), TOKEN_TYPE.FUNCTION_CALL)
  
  @staticmethod
  def build_if_code_token(start_line: Line, initial_code: CodeBlock) -> CodeToken|None:
    node_content = []
    open_braces_count = 0
    idx = 0
    line = start_line
    while(True):
      node_content.append(line)
      if line.is_open_brace():
        open_braces_count+=1
      if line.is_close_brace():
        open_braces_count-=1
      if len(node_content) == 1 and not line.is_open_brace():
        idx+=1
        next_next_line = initial_code.get(idx)
        if not next_next_line:
          return None # for now we don't support if(a) foo()s
        if next_next_line.is_open_brace():
          open_braces_count+=1
          node_content.append(next_next_line)
        else:
          return CodeToken.build_one_line_if_code_block(start_line, initial_code)
      if open_braces_count == 0:
        idx+=1
        next_next_line = initial_code.get(idx)
        if not next_next_line or not next_next_line.content.startswith('else'):
          break
        node_content.append(next_next_line)
      idx+=1
      line = initial_code.get(idx)
      if not line:
        print("TODO NOT LINE")
        break
    return CodeToken(start_line, CodeBlock(node_content), TOKEN_TYPE.IF)

  @staticmethod
  def build_one_line_if_code_block(start_line: Line, initial_code: CodeBlock) -> CodeToken|None:
    """
      if(a) foo()
      For now - we can avoid this. Think about it in a future!
    """
    node_content = []
    line = start_line
    node_content.append(line)
    idx = 1
    next_line = initial_code.get(idx)
    if not next_line:
      print("IF NONE ONE LINE TODO")
      return None
    node_content.append(next_line)
    idx+=1
    else_line = initial_code.get(idx)
    if else_line and else_line.startswith('else'):
      node_content.append(else_line)
      idx+=1
      else_next_line = initial_code.get(idx)
      if not else_next_line:
        print("ELSE ONT LINE NONE TODO!")
        return None
      node_content.append(else_next_line)
    currentToken = CodeToken(start_line, CodeBlock(node_content), TOKEN_TYPE.IF)
    return currentToken
  
  @staticmethod
  def build_variable_declaration(start_line: Line, initial_code: CodeBlock) -> CodeToken|None:
    pattern = r'^\b(?:int|float|double|char|long|short)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*;'
    match = re.search(pattern, start_line.content)
    if match:
      return CodeToken(start_line, CodeBlock([start_line]), TOKEN_TYPE.VARIABLE_DECLARATION)
    return None
  
  @staticmethod
  def build_variable_assignment(start_line: Line, initial_code: CodeBlock) -> CodeToken|None:
    pattern = r'^\b(?:int|float|double|char|long|short)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^;]*;'
    match = re.search(pattern, start_line.content)
    if match:
      return CodeToken(start_line, CodeBlock([start_line]), TOKEN_TYPE.VARIABLE_ASSIGNMENT)
    return None
  
  @staticmethod
  def build_variable_reassignment(start_line: Line, initial_code: CodeBlock) -> CodeToken|None:
    pattern = r'^(?!\s*(?:int|float|double|char|long|short)\s)\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^;]*;'
    match = re.search(pattern, start_line.content)
    if match:
      return CodeToken(start_line, CodeBlock([start_line]), TOKEN_TYPE.VARIABLE_REASSIGNMENT)
    return None
  
  @staticmethod
  def generate_new_code_token(
    content: list[Line],
    type: TOKEN_TYPE
  ) -> CodeToken:
    return CodeToken(content[0], CodeBlock(content), type)

  def print(self, debug=False) -> str:
    result = f"id: {self.id}\n"
    result += f"type: {self.type}\n"
    result += f"line_start: {self.line.content}\n"
    result += f"children: {list(map(lambda x: x.id, self.children))}\n"

    if debug:
      result += f"content: \n {self.print_content()}\n"
    return result