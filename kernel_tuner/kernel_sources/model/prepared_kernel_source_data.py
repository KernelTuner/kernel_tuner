from dataclasses import dataclass
from typing import Any


@dataclass
class PreparedKernelSourceData:
    temp_files: Any
    kernel_name: str
    kernel_str: Any
    kernel_fn: Any