import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import kernel_tuner.cuda as cuda
import kernel_tuner.opencl as opencl
import kernel_tuner.interface as kernel_tuner

from nose import SkipTest
from nose.tools import nottest

@nottest
def skip_if_no_cuda_device():
    try:
        from pycuda.autoinit import context
    except (ImportError, Exception) as e:
        raise SkipTest("PyCuda not installed")

@nottest
def skip_if_no_opencl():
    try:
        import pyopencl
    except ImportError as e:
        raise SkipTest("PyOpenCL not installed")

