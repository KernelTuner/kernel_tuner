import numpy
from nose.tools import nottest

try:
    from mock import patch, Mock
except ImportError:
    from unittest.mock import patch, Mock

from kernel_tuner import cuda

def setup_mock(drv):
    context = Mock()
    devprops = { 'MAX_THREADS_PER_BLOCK': 1024,
                'COMPUTE_CAPABILITY_MAJOR': 5,
                'COMPUTE_CAPABILITY_MINOR': 5 }
    context.return_value.get_device.return_value.get_attributes.return_value = devprops
    drv.Device.return_value.make_context.return_value = context()
    drv.mem_alloc.return_value = 'mem_alloc'
    return drv

@patch('kernel_tuner.cuda.drv')
def test_ready_argument_list(drv):
    drv = setup_mock(drv)

    size = 5
    a = numpy.int32(75)
    b = numpy.random.randn(size).astype(numpy.float32)
    arguments = [a, b]

    dev = cuda.CudaFunctions(0)
    gpu_args = dev.ready_argument_list(arguments)

    print(drv.mock_calls)
    print(gpu_args)

    drv.mem_alloc.assert_called_once_with(20)
    drv.memcpy_htod.assert_called_once_with('mem_alloc', b)

    assert isinstance(gpu_args[0], numpy.int32)

@patch('kernel_tuner.cuda.SourceModule')
@patch('kernel_tuner.cuda.drv')
def test_compile(drv, src_mod):
    drv = setup_mock(drv)

    src_mod.return_value.get_function.return_value = 'func'

    dev = cuda.CudaFunctions(0)
    kernel_string = "__global__ void vector_add()"
    func = dev.compile("vector_add", kernel_string)

    assert src_mod.call_count == 1
    assert dev.current_module is src_mod.return_value
    assert func == 'func'

    assert kernel_string == list(src_mod.mock_calls[0])[1][0]
    optional_args = list(src_mod.mock_calls[0])[2]
    assert optional_args['code'] == 'sm_55'
    assert optional_args['arch'] == 'compute_55'


@nottest
def test_func(a, b, block=0, grid=0):
    pass

@patch('kernel_tuner.cuda.drv')
def test_benchmark(drv):
    drv = setup_mock(drv)

    drv.Event.return_value.time_since.return_value = 0.1

    dev = cuda.CudaFunctions(0)
    args = [1, 2]
    time = dev.benchmark(test_func, args, (1,2), (1,2))
    assert time > 0

    assert dev.context.synchronize.call_count == 2*dev.iterations
    assert drv.Event.return_value.record.call_count == 2*dev.iterations
    assert drv.Event.return_value.time_since.call_count == dev.iterations


@patch('kernel_tuner.cuda.drv')
def test_copy_constant_memory_args(drv):
    drv = setup_mock(drv)

    fake_array = numpy.zeros(10).astype(numpy.float32)
    cmem_args = { 'fake_array': fake_array }

    dev = cuda.CudaFunctions(0)
    dev.current_module = Mock()
    dev.current_module.get_global.return_value = ['get_global']

    dev.copy_constant_memory_args(cmem_args)

    drv.memcpy_htod.assert_called_once_with('get_global', fake_array)
    dev.current_module.get_global.assert_called_once_with('fake_array')

