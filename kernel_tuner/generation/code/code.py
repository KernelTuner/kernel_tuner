from __future__ import annotations
from kernel_tuner.generation.code.line import Line

class Code:
    
  def __init__(self, lines: list[str]) -> None:
    self.initial_lines = list(map(lambda x: Line(x[1], x[0]), enumerate(lines.copy())))
    self.lines = list(map(lambda x: Line(x[1], x[0]), enumerate(lines)))
    self.num_lines = len(self.lines)

  def print(self):
    print(self.to_text())

  def to_text(self) -> str:
    return '\n'.join(list(map(lambda x: x.content, self.initial_lines)))

class CodeBlock:
    
    def __init__(self, lines: list[Line]):
      self.lines = lines

    def size(self) -> int:
      return len(self.lines)
    
    def get(self, i) -> Line|None:
      if i < self.size():
        return self.lines[i]
      else:
        return None