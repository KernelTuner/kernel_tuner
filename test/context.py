import sys
import pytest

try:
    import pycuda.driver as drv
    drv.init()
    cuda_present=True
except:
    cuda_present=False

try:
    import pyopencl
    opencl_present=True
    if 'namespace' in str(sys.modules['pyopencl']):
        opencl_present=False
    if len(pyopencl.get_platforms())==0:
        opencl_present=False
except:
    opencl_present=False

try:
    import noodles
    noodles_present=True
except:
    noodles_present=False

skip_if_no_cuda=pytest.mark.skipif(not cuda_present,
                    reason="PyCuda not installed or no CUDA device detected")
skip_if_no_opencl=pytest.mark.skipif(not opencl_present,
                    reason="PyCuda not installed or no CUDA device detected")
skip_if_no_noodles=pytest.mark.skipif(not noodles_present,
                    reason="PyCuda not installed or no CUDA device detected")

