from kernel_tuner.generation.token.pragma_token import *
from typing import TypeAlias

PragmaTuneParams: TypeAlias = list[tuple[PRAGMA_KEYWORDS, str, list[str]]]

def convertPragmaTuneToDict(pragmaTuneParams: PragmaTuneParams) -> dict[str, list[str]]:
  return dict(map(lambda x: (x[1], x[2]), pragmaTuneParams))

def filter_pragmas_by_type(pragmas: list[PragmaToken], type: PRAGMA_TOKEN_TYPE) -> list[PragmaToken]:
  return list(filter(lambda x: x.pragma_type == type, pragmas))

def filter_pragmas_contains_keyword(
    pragmas: list[PragmaToken], 
    contains_keywords: list[PRAGMA_KEYWORDS],
    exclude_keywords: list[PRAGMA_KEYWORDS] = []
) -> list[PragmaToken]:
  resulst = []
  for pragma in pragmas:
    if (any(x in contains_keywords for x in pragma.keywords) and all(x not in exclude_keywords for x in pragma.keywords) ):
        resulst.append(pragma)
  return resulst