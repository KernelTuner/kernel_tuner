from enum import Enum

class Language(Enum):
    CUDA = "CUDA"
    OpenCL = "OPENCL"
    C = "C"
    HIP = "HIP"
    TRITON = "TRITON"
    FORTRAN = "FORTRAN"
    NVCUDA = "NVCUDA"