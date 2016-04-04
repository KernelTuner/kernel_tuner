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
    #except pycuda.driver.RuntimeError, e:
    except Exception, e:
        if "No module named pycuda.autoinit" in str(e):
            raise SkipTest("PyCuda not installed")
        elif "no CUDA-capable device is detected" in str(e):
            raise SkipTest("no CUDA-capable device is detected")
        else:
            raise e

@nottest
def skip_if_no_opencl():
    try:
        import pyopencl
    except Exception, e:
        if "No module named pyopencl" in str(e):
            raise SkipTest("PyOpenCL not installed")
        else:
            raise e

