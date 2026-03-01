from enum import Enum

class Language(Enum):
    CUDA = "CUDA"
    OPENCL = "OPENCL"
    C = "C"
    HIP = "HIP"
    FORTRAN = "FORTRAN"
    NVCUDA = "NVCUDA"
    GENERIC_PYTHON = "GENERIC_PYTHON"
    CUPY = "CUPY"
    HYPERTUNER = "HYPERTUNER"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """
        Test if a language is valid in Kernel Tuner framework.
        """
        return value in cls._value2member_map_