from __future__ import print_function

import numpy
import kernel_tuner

def test_random_sample():

    kernel_string = "float test_kernel(float *a) { return 1.0f; }"
    a = numpy.array([1,2,3]).astype(numpy.float32)

    tune_params = {"block_size_x": range(1,25)}
    print(tune_params)

    result, _ = kernel_tuner.tune_kernel("test_kernel", kernel_string, (1,1), [a], tune_params, sample=True)

    print(result)

    #check that number of benchmarked kernels is 10% (rounded up)
    assert len(result) == 3

    #check all returned results make sense
    for v in result:
        assert v['time'] == 1.0

