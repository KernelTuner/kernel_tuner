from __future__ import annotations

class Line:
  def __init__(self, line, line_number):
    self.content = line.strip()
    self.line_number = line_number
    self.words = self.content.split()

  def replace(self, remove: str, append: str = ''):
    self.content = self.content.replace(remove, append).strip()

  def is_open_brace(self) -> bool:
    return self.content.startswith('{') or self.content.endswith('{')

  def is_close_brace(self) -> bool:
    return self.content.startswith('}') or self.content.endswith('}')
  
  def startswith(self, start) -> bool:
    return self.content.startswith(start)
  
  def endswith(self, start) -> bool:
    return self.content.endswith(start)
  
  def append(self, line: Line, split: str):
    self.content += split + line.content
    self.content.strip()

  def find(self, finder):
    if finder in self.content:
      return self.content.find(finder)
    else:
      return None
    
  def len(self):
    return len(self.content)
    
  def find_end_line(self):
    return self.find(';')

  def __add__(self, line) -> Line:
    self.content += ' ' + line.content
    self.content.strip()
    return self

def merge_to_one_line(lines: list[Line]) -> str:
    return ''.join(list(map(lambda x: x.content, lines)))

