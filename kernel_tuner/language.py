from enum import Enum


class Language(str, Enum):
    """Languages (and explicitly selectable backends) supported by Kernel Tuner.

    Members are also strings, so code that compares ``lang`` to plain strings
    (e.g. ``lang.upper() == "CUDA"``) keeps working.
    """

    CUDA = "CUDA"
    PYCUDA = "PYCUDA"
    CUPY = "CUPY"
    NVCUDA = "NVCUDA"
    OPENCL = "OPENCL"
    HIP = "HIP"
    C = "C"
    FORTRAN = "FORTRAN"
    JULIA = "JULIA"
    HYPERTUNER = "HYPERTUNER"
    GENERIC_PYTHON = "GENERIC_PYTHON"

    def __str__(self):
        return self.value

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Test if a language is valid in Kernel Tuner framework."""
        return value in cls._value2member_map_
