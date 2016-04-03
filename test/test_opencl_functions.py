import numpy
from nose import SkipTest
from nose.tools import nottest, raises
from .context import kernel_tuner
from .context import opencl

try:
    import pyopencl
except Exception:
    pass

@nottest
def skip_if_no_opencl():
    try:
        import pyopencl
    except Exception, e:
        if "No module named pyopencl" in str(e):
            raise SkipTest("PyOpenCL not installed")
        else:
            raise e

def test_create_gpu_args():

    skip_if_no_opencl()

    size = 1000
    a = numpy.int32(75)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)

    arguments = [c, a, b]

    dev = opencl.OpenCLFunctions(0)
    gpu_args = dev.create_gpu_args(arguments)

    assert isinstance(gpu_args[0], pyopencl.Buffer)
    assert isinstance(gpu_args[1], numpy.int32)
    assert isinstance(gpu_args[2], pyopencl.Buffer)

    gpu_args[0].release()
    gpu_args[2].release()

def test_compile():

    skip_if_no_opencl()

    original_kernel = """
    __kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g) {
        int gid = get_global_id(0);
        __local float test[shared_size];
        test[0] = a_g[gid];
        res_g[gid] = test[0] + b_g[gid];
    }
    """

    kernel_string = original_kernel.replace("shared_size", str(100*1024*1024))

    dev = opencl.OpenCLFunctions(0)
    func = dev.compile("sum", kernel_string)

    assert isinstance(func, pyopencl.Kernel)
