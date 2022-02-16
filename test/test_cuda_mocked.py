import numpy as np

try:
    from mock import patch, Mock
except ImportError:
    from unittest.mock import patch, Mock

from kernel_tuner import cuda
from kernel_tuner.core import KernelSource, KernelInstance


def setup_mock(drv):
    context = Mock()
    devprops = {'MAX_THREADS_PER_BLOCK': 1024,
                'COMPUTE_CAPABILITY_MAJOR': 5,
                'COMPUTE_CAPABILITY_MINOR': 5}
    context.return_value.get_device.return_value.get_attributes.return_value = devprops
    context.return_value.get_device.return_value.compute_capability.return_value = "55"
    drv.Device.return_value.make_context.return_value = context()
    drv.mem_alloc.return_value = 'mem_alloc'
    return drv


@patch('kernel_tuner.cuda.nvml')
@patch('kernel_tuner.cuda.DynamicSourceModule')
@patch('kernel_tuner.cuda.drv')
def test_ready_argument_list(drv, *args):
    drv = setup_mock(drv)

    size = 5
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    arguments = [a, b]

    with cuda.CudaFunctions(0) as dev:
        gpu_args = dev.ready_argument_list(arguments)

    print(drv.mock_calls)
    print(gpu_args)

    drv.mem_alloc.assert_called_once_with(20)
    drv.memcpy_htod.assert_called_once_with('mem_alloc', b)

    assert isinstance(gpu_args[0], np.int32)


@patch('kernel_tuner.cuda.nvml')
@patch('kernel_tuner.cuda.DynamicSourceModule')
@patch('kernel_tuner.cuda.drv')
def test_compile(drv, *args):

    # setup mocked stuff
    drv = setup_mock(drv)
    with cuda.CudaFunctions(0) as dev:
        dev.source_mod = Mock()
        dev.source_mod.return_value.get_function.return_value = 'func'

        # call compile
        kernel_string = "__global__ void vector_add()"
        kernel_name = "vector_add"
        kernel_sources = KernelSource(kernel_name, kernel_string, "cuda")
        kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])
        func = dev.compile(kernel_instance)

        # verify behavior
        assert dev.source_mod.call_count == 1
        assert dev.current_module is dev.source_mod.return_value
        assert func == 'func'

        assert kernel_string == list(dev.source_mod.mock_calls[0])[1][0]
        optional_args = list(dev.source_mod.mock_calls[0])[2]
        assert optional_args['code'] == 'sm_55'
        assert optional_args['arch'] == 'compute_55'


def dummy_func(a, b, block=0, grid=0, shared=0, stream=None, texrefs=None):
    pass


@patch('kernel_tuner.cuda.nvml')
@patch('kernel_tuner.cuda.DynamicSourceModule')
@patch('kernel_tuner.cuda.drv')
def test_benchmark(drv, *args):
    drv = setup_mock(drv)

    drv.Event.return_value.time_since.return_value = 0.1

    with cuda.CudaFunctions(0) as dev:
        res = dev.benchmark(dummy_func, [1, 2], (1, 2), (1, 2))
        assert res["time"] > 0

        assert dev.context.synchronize.call_count > 1
        assert drv.Event.return_value.synchronize.call_count == dev.iterations
        assert drv.Event.return_value.record.call_count == 2*dev.iterations
        assert drv.Event.return_value.time_since.call_count == dev.iterations


@patch('kernel_tuner.cuda.nvml')
@patch('kernel_tuner.cuda.DynamicSourceModule')
@patch('kernel_tuner.cuda.drv')
def test_copy_constant_memory_args(drv, *args):
    drv = setup_mock(drv)

    fake_array = np.zeros(10).astype(np.float32)
    cmem_args = {'fake_array': fake_array}

    with cuda.CudaFunctions(0) as dev:
        dev.current_module = Mock()
        dev.current_module.get_global.return_value = ['get_global']

        dev.copy_constant_memory_args(cmem_args)

        drv.memcpy_htod.assert_called_once_with('get_global', fake_array)
        dev.current_module.get_global.assert_called_once_with('fake_array')


@patch('kernel_tuner.cuda.nvml')
@patch('kernel_tuner.cuda.DynamicSourceModule')
@patch('kernel_tuner.cuda.drv')
def test_copy_texture_memory_args(drv, *args):
    drv = setup_mock(drv)

    fake_array = np.zeros(10).astype(np.float32)
    texmem_args = {'fake_tex': fake_array}

    texref = Mock()

    with cuda.CudaFunctions(0) as dev:
        dev.current_module = Mock()
        dev.current_module.get_texref.return_value = texref

        dev.copy_texture_memory_args(texmem_args)

        drv.matrix_to_texref.assert_called_once_with(fake_array, texref, order="C")
        dev.current_module.get_texref.assert_called_once_with('fake_tex')

        texmem_args = {'fake_tex2': {'array': fake_array, 'filter_mode': 'linear', 'address_mode': ['border', 'clamp']}}

        dev.copy_texture_memory_args(texmem_args)
        drv.matrix_to_texref.assert_called_with(fake_array, texref, order="C")
        dev.current_module.get_texref.assert_called_with('fake_tex2')
        texref.set_filter_mode.assert_called_once_with(drv.filter_mode.LINEAR)
        texref.set_address_mode.assert_any_call(0, drv.address_mode.BORDER)
        texref.set_address_mode.assert_any_call(1, drv.address_mode.CLAMP)
