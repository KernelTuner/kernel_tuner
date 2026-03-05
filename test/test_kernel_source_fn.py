import os
from pathlib import Path
import pytest
import inspect
import ast
import functools # for an example decorator
from kernel_tuner.kernel_sources.kernel_source import KernelSource
# Note: need subclass imports to test for types and instantiate subclasses direclty.
# Normally, just importing KernelSource is enough.
from kernel_tuner.kernel_sources.kernel_source_fn import KernelSourceFn
from kernel_tuner.kernel_sources.kernel_source_str import KernelSourceStr

KS_FILE = Path(__file__).resolve()
# Helper functions --------------------------------------------

def normalize_ast(src: str):
    return ast.dump(ast.parse(src), include_attributes=False)

def mock_kernel(a, b):
    mock_param = 256
    if a < mock_param:
        b = 42
    return mock_param

def kernel_with_kwarg(a, mock_param):
    return mock_param

# NOTE should kernel_with_kwarg now return the tuning param value for mock param or 5?
def kernel_with_dependency():
    mock_param = 42
    foo = kernel_with_kwarg(mock_param, 5)
    return mock_kernel(42, 42)

def call_mock(kernel_function, args, kwargs, grid, threads, params):
        kernel_function(*args, **kwargs)

class TilusLike():
    def __init__(self):
        self.mock_param = 32  
    
    def __call__(self):
        return self.mock_param

    def another_function(self):
        return self.mock_param

@functools.lru_cache()   
def kernel_with_decorator():
    mock_param = 42
    return mock_param
    

# Tests ------------------------------------------------------

def test_factory_behaviour():
    # KernelSourceFn should only be created when language generic_python is supplied
    ks_fn = KernelSource("mock_kernel", KS_FILE, "generic_python", call_function=call_mock)
    ks_str = KernelSource("vector_add", 'extern "C" __global__ void vector_add(float *c, float *a, float *b, int n) {', lang=None)
    
    assert isinstance(ks_fn, KernelSourceFn)
    assert isinstance(ks_str, KernelSourceStr)


def test_initiation():
    '''
    Test invalid KernelSourceFn initations    
    '''
    with pytest.raises(ValueError, match=r"call_function must be supplied for language .*"):
        KernelSource("mock_kernel", KS_FILE, "generic_python")
    
    with pytest.raises(FileNotFoundError, match=r".* No such file or directory: .*"):
        KernelSource("mock_kernel", "This is a string Kernel", "generic_python", call_function=call_mock)

    with pytest.raises(TypeError, match="Error kernel_source does not specify a path to a file"):
        KernelSource("mock_kernel", mock_kernel, "generic_python", call_function=call_mock)

    with pytest.raises(ValueError, match=r"KernelSourceFn only supports a single kernel source"):
        KernelSource("mock_kernel", [KS_FILE, "another file"], "generic_python", call_function=call_mock)

    with pytest.raises(TypeError, match=r".* is not a callable object"):
        KernelSource("mock_kernel", KS_FILE, "generic_python", call_function="not a function")



def test_param_subsitution():
    params = {"mock_param": 128}
    ks = KernelSourceFn("mock_kernel", KS_FILE, "generic_python", call_function=call_mock)
    new_kernel_fn, _ = ks.apply_params_to_source_fn(params)

    actual_src = inspect.getsource(new_kernel_fn)
    expected_src = """
def mock_kernel(a, b):
    mock_param = 128
    if a < 128:
        b = 42
    return 128
"""

    assert normalize_ast(actual_src) == normalize_ast(expected_src)


def test_imports():
    '''
    Import statements that are present in the file where the function lives
    should also be present in the file where the new function lives. 
    '''

    params = {"mock_param": 128}
    ks = KernelSourceFn("mock_kernel", KS_FILE, "generic_python", call_function=call_mock)
    _, temp_path = ks.apply_params_to_source_fn(params)

    # Check if imports are present
    with open(temp_path) as f:
        full_src = f.read()

    assert "import pytest" in full_src
    assert "import inspect" in full_src


def test_param_substitution_class():
    '''
    Tilus uses the __call__ function of a class to define its kernel. Therefore,
    param substitution should also work classes. Param substiution in other class
    functions is also supported.
    '''

    params = {"mock_param": 128}
    ks = KernelSourceFn("TilusLike", KS_FILE, "generic_python", call_function=call_mock)

    new_kernel_fn, _ = ks.apply_params_to_source_fn(params)

    actual_src = inspect.getsource(new_kernel_fn)
    expected_src = """
class TilusLike():
    def __init__(self):
        self.mock_param = 128  

    def __call__(self):
        return 128
    
    def another_function(self):
        return 128
"""
    assert normalize_ast(actual_src) == normalize_ast(expected_src)



def test_decorator():
    params = {"mock_param": 128}
    ks = KernelSourceFn("kernel_with_decorator", KS_FILE, "generic_python", call_function=call_mock)
    new_kernel_fn, _ = ks.apply_params_to_source_fn(params)
    actual_src = inspect.getsource(new_kernel_fn)
    expected_src = """
@functools.lru_cache()   
def kernel_with_decorator():
    mock_param = 128
    return 128
"""
    assert hasattr(new_kernel_fn, "__wrapped__")
    assert normalize_ast(actual_src) == normalize_ast(expected_src)


def test_dependencies():
    params = {"mock_param": 128}
    ks = KernelSourceFn("kernel_with_dependency", KS_FILE, "generic_python", call_function=call_mock)
    new_kernel_fn, _ = ks.apply_params_to_source_fn(params)
    res = new_kernel_fn() # This should not throw an error if the dependency exists in the module.

    assert res == 128



