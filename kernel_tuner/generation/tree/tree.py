from kernel_tuner.generation.token.pragma_token import PragmaToken
from kernel_tuner.generation.token.token import *
from kernel_tuner.generation.token.code_token import *
from kernel_tuner.generation.code.line import Line
from kernel_tuner.generation.code.code import Code, CodeBlock
from kernel_tuner.util import write_file

class Tree:

  def __init__(self, root: PragmaToken, pragma_tokens: list[PragmaToken]) -> None:    
    self.root = root
    self.root.pragma_type = PRAGMA_TOKEN_TYPE.ROOT
    self.pragma_tokens = pragma_tokens

  def append_node(self, parent: Token, child: Token, insert_token: Token):
    if child not in parent.children:
      print("ERRPR PARENT NOT REAL PARENT!")
      return
    parent.remove_child(child)
    parent.append_child(insert_token)
    insert_token.append_child(child)


  def replace_node(self, replace_token: Token, new_token: Token):
    parent = replace_token.parent
    if not parent:
      return
    parent.remove_child(replace_token)
    parent.append_child(new_token)

  def dfs(self, func, *args):
    self.__dfs_loop(self.root, func, *args)

  def __dfs_loop(self, node, func, *args):
    func(node, *args)
    for children in node.pragma_children:
      self.__dfs_loop(children, func, *args)


  def dfs_print(self, node = None, debug_file_name=None):
    cur_node = node if node else self.root
    if debug_file_name:
      write_file(debug_file_name, cur_node.print(True) + '\n', "a")
    else:
      print(cur_node.print(True))
    for child in cur_node.children:
      self.dfs_print(child, debug_file_name)
    


class TreeBuilder:

  def __init__(self, code: Code) -> None:
    self.tree_root = PragmaToken(Line('', 0), 0)
    self.node_map = {0: self.tree_root}
    self.current_level = 1
    self.current_bracket_level = 0
    self.last_bracket_level = 0
    self.code = code
    self.pragma_tokens = []
    self.idx = 0

  def build_tree(self):
    while(self.idx < self.code.num_lines):
      line = self.code.lines[self.idx]
      if line.is_open_brace():
        self.current_bracket_level+=1
      if line.is_close_brace():
        self.current_bracket_level-=1

      if line.startswith("#pragma omp"):
        node = self.__build_pragma_token(line)

        if not node.level - 1 in self.node_map:
          print("ERROR!!")
          exit(0)
        
        self.node_map[node.level - 1].append_child(node)
        self.node_map[node.level] = node

        self.pragma_tokens.append(node)

      self.idx+=1

    self.__build_code_token()
    return Tree(self.tree_root, self.pragma_tokens)


  def __build_pragma_token(self, line) -> PragmaToken:
    if self.code.lines[self.idx-1].startswith('#pragma omp') and not self.pragma_tokens[len(self.pragma_tokens)-1].type.is_data():
      self.current_level+=1
    else:
      if self.current_bracket_level > self.last_bracket_level:
        self.current_level+=1
        self.last_bracket_level = self.current_bracket_level
      elif self.current_bracket_level < self.last_bracket_level:
        self.current_level-=1
        self.last_bracket_level = self.current_bracket_level

      line = line
      if line.endswith('\\'):
        while(True):
          line.replace('\\')
          self.idx+=1
          next_line = self.code.lines[self.idx]
          line += next_line
          if not next_line.endswith('\\'):            
            break          
        self.idx-=1

    node = PragmaToken(line, self.current_bracket_level)
    return node


  def __build_code_token(self):
    for node in self.pragma_tokens:
      if node.pragma_type == PRAGMA_TOKEN_TYPE.ROOT:
        continue
      idx = node.line.line_number + 1
      if idx >= self.code.num_lines:
        return
      next_line = self.code.initial_lines[idx]
      if next_line.startswith('#pragma omp'):
        continue
      code_token = CodeToken.build_code_token(next_line, CodeBlock(self.code.initial_lines[idx:]))
      if code_token:
        node.append_child(code_token)
      
