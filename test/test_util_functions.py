from __future__ import print_function

from collections import OrderedDict
import os
import json
import warnings

import numpy as np
import pytest
from pytest import raises

from .context import skip_if_no_cuda, skip_if_no_opencl

from kernel_tuner.interface import Options
import kernel_tuner.core as core
import kernel_tuner.cuda as cuda
import kernel_tuner.opencl as opencl
from kernel_tuner.util import *

block_size_names = ["block_size_x", "block_size_y", "block_size_z"]


def test_get_grid_dimensions1():
    problem_size = (1024, 1024, 1)
    params = {"block_x": 41, "block_y": 37}

    grid_div = (["block_x"], ["block_y"], None)

    grid = get_grid_dimensions(problem_size, params, grid_div, block_size_names)

    assert len(grid) == 3
    assert isinstance(grid[0], int)
    assert isinstance(grid[1], int)

    assert grid[0] == 25
    assert grid[1] == 28
    assert grid[2] == 1

    grid = get_grid_dimensions(problem_size, params, (grid_div[0], None, None), block_size_names)

    assert grid[0] == 25
    assert grid[1] == 1024
    assert grid[2] == 1

    grid = get_grid_dimensions(problem_size, params, (None, grid_div[1], None), block_size_names)

    assert grid[0] == 1024
    assert grid[1] == 28
    assert grid[2] == 1

    grid = get_grid_dimensions(problem_size, params,
                    (None, lambda p: p["block_x"], lambda p: p["block_y"]*p["block_x"]), block_size_names)

    assert grid[0] == 1024
    assert grid[1] == 25
    assert grid[2] == 1


def test_get_grid_dimensions2():
    problem_size = (1024, 1024, 1)
    params = {"block_x": 41, "block_y": 37}

    grid_div_x = ["block_x*8"]
    grid_div_y = ["(block_y+2)/8"]

    grid = get_grid_dimensions(problem_size, params, (grid_div_x, grid_div_y, None), block_size_names)

    assert grid[0] == 4
    assert grid[1] == 256


def test_get_grid_dimensions3():
    problem_size = (1024, 1024, 1)
    params = {"block_x": 41, "block_y": 37}

    grid_div_x = ["block_x", "block_y"]
    grid_div_y = ["(block_y+2)/8"]

    def assert_grid_dimensions(problem_size):
        grid = get_grid_dimensions(problem_size, params,
                                   (grid_div_x, grid_div_y, None), block_size_names)
        assert grid[0] == 1
        assert grid[1] == 256
        assert grid[2] == 1

    assert_grid_dimensions(problem_size)

    problem_size = (np.int32(1024), np.int64(1024), 1)
    assert_grid_dimensions(problem_size)


def test_get_problem_size1():
    problem_size = ("num_blocks_x", "num_blocks_y*3")
    params = {"num_blocks_x": 71, "num_blocks_y": 57}

    answer = get_problem_size(problem_size, params)
    assert answer[0] == 71
    assert answer[1] == 171
    assert answer[2] == 1


def test_get_problem_size2():
    problem_size = "num_blocks_x"
    params = {"num_blocks_x": 71}

    answer = get_problem_size(problem_size, params)
    assert answer[0] == 71
    assert answer[1] == 1
    assert answer[2] == 1


def test_get_problem_size3():
    with raises(TypeError):
        problem_size = (3.8, "num_blocks_y*3")
        params = {"num_blocks_y": 57}
        get_problem_size(problem_size, params)


def test_get_problem_size4():
    params = {"num_blocks_x": 71}

    answer = get_problem_size(lambda p: (p["num_blocks_x"], 1, 13), params)
    assert answer[0] == 71
    assert answer[1] == 1
    assert answer[2] == 13


def test_get_thread_block_dimensions():

    params = {"block_size_x": 123, "block_size_y": 257}

    threads = get_thread_block_dimensions(params)
    assert len(threads) == 3
    assert isinstance(threads[0], int)
    assert isinstance(threads[1], int)
    assert isinstance(threads[2], int)

    assert threads[0] == 123
    assert threads[1] == 257
    assert threads[2] == 1


def test_prepare_kernel_string():
    kernel = "this is a weird kernel"
    params = dict()
    params["is"] = 8

    _, output = prepare_kernel_string("this", kernel, params, (3, 7), (1, 2, 3), block_size_names, "")
    expected = "#define kernel_tuner 1\n" \
               "#define is 8\n" \
               "#define block_size_z 3\n" \
               "#define block_size_y 2\n" \
               "#define block_size_x 1\n" \
               "#define grid_size_y 7\n" \
               "#define grid_size_x 3\n" \
               "this is a weird kernel"
    assert output == expected


def test_replace_param_occurrences():
    kernel = "this is a weird kernel"
    params = dict()
    params["is"] = 8
    params["weird"] = 14

    new_kernel = replace_param_occurrences(kernel, params)
    assert new_kernel == "th8 8 a 14 kernel"

    new_kernel = replace_param_occurrences(kernel, dict())
    assert kernel == new_kernel

    params = dict()
    params["blablabla"] = 8
    new_kernel = replace_param_occurrences(kernel, params)
    assert kernel == new_kernel


def test_check_restrictions():
    params = {"a": 7, "b": 4, "c": 3}
    print(params.values())
    print(params.keys())
    restrictions = [["a==b+c"], ["a==b+c", "b==b", "a-b==c"],
                    ["a==b+c", "b!=b", "a-b==c"],
                    lambda p:p["a"] == p["b"] + p["c"]]
    expected = [True, True, False, True]
    # test the call returns expected
    for r, e in zip(restrictions, expected):
        answer = check_restrictions(r, params.values(), params.keys(), False)
        print(answer)
        assert answer == e


def test_detect_language1():
    kernel_string = "__global__ void vector_add( ... );"
    lang = detect_language(kernel_string)
    assert lang == "CUDA"


def test_detect_language2():
    kernel_string = "__kernel void vector_add( ... );"
    lang = detect_language(kernel_string)
    assert lang == "OpenCL"


def test_detect_language3():
    kernel_string = "blabla"
    lang = detect_language(kernel_string)
    assert lang == "C"


@skip_if_no_cuda
def test_get_device_interface1():
    lang = "CUDA"
    with core.DeviceInterface(core.KernelSource("", "", lang=lang)) as dev:
        assert isinstance(dev, core.DeviceInterface)
        assert isinstance(dev.dev, cuda.CudaFunctions)


@skip_if_no_opencl
def test_get_device_interface2():
    lang = "OpenCL"
    with core.DeviceInterface(core.KernelSource("", "", lang=lang)) as dev:
        assert isinstance(dev, core.DeviceInterface)
        assert isinstance(dev.dev, opencl.OpenCLFunctions)


def test_get_device_interface3():
    with raises(Exception):
        lang = "blabla"
        with core.DeviceInterface(lang) as dev:
            pass


def assert_user_warning(f, args, substring=None):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f(*args)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        if substring:
            assert substring in str(w[-1].message)


def assert_no_user_warning(f, args):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f(*args)
        assert len(w) == 0


def test_check_argument_list1():
    kernel_name = "test_kernel"
    kernel_string = """__kernel void test_kernel(int number, char * message, int * numbers) {
    numbers[get_global_id(0)] = numbers[get_global_id(0)] * number;
    }
    """
    args = [np.int32(5), 'blah', np.array([1, 2, 3])]
    try:
        check_argument_list(kernel_name, kernel_string, args)
        print("Expected a TypeError to be raised")
        assert False
    except TypeError as e:
        print(str(e))
        assert "at position 1" in str(e)
    except Exception:
        print("Expected a TypeError to be raised")
        assert False


def test_check_argument_list2():
    kernel_name = "test_kernel"
    kernel_string = """__kernel void test_kernel
        (char number, double factors, int * numbers, const unsigned long * moreNumbers) {
        numbers[get_global_id(0)] = numbers[get_global_id(0)] * factors[get_global_id(0)] + number;
        }
        """
    args = [np.byte(5), np.float64(4.6), np.int32([1, 2, 3]), np.uint64([3, 2, 111])]
    assert_no_user_warning(check_argument_list, [kernel_name, kernel_string, args])


def test_check_argument_list3():
    kernel_name = "test_kernel"
    kernel_string = """__kernel void test_kernel (__global const ushort number, __global half * factors, __global long * numbers) {
        numbers[get_global_id(0)] = numbers[get_global_id(0)] * factors[get_global_id(0)] + number;
        }
        """
    args = [np.uint16(42), np.float16([3, 4, 6]), np.int32([300])]
    assert_user_warning(check_argument_list, [kernel_name, kernel_string, args], "at position 2")


def test_check_argument_list4():
    kernel_name = "test_kernel"
    kernel_string = """__kernel void test_kernel(__global const ushort number, __global half * factors, __global long * numbers) {
        numbers[get_global_id(0)] = numbers[get_global_id(0)] * factors[get_global_id(0)] + number;
        }
        """
    args = [np.uint16(42), np.float16([3, 4, 6]), np.int64([300]), np.ubyte(32)]
    assert_user_warning(check_argument_list, [kernel_name, kernel_string, args], "do not match in size")


def test_check_argument_list5():
    kernel_name = "my_test_kernel"
    kernel_string = """ //more complicated test function(because I can)

        __device__ float some_lame_device_function(float *a) {
            return a[0];
        }

        __global__ void my_test_kernel(double *a,
                                       float *b, int c,
                                       int d) {

            a[threadIdx.x] = b[blockIdx.x]*c*d;
        }
        """
    args = [np.array([1, 2, 3]).astype(np.float64),
            np.array([1, 2, 3]).astype(np.float32),
            np.int32(6), np.int32(7)]
    assert_no_user_warning(check_argument_list, [kernel_name, kernel_string, args])


def test_check_argument_list6():
    kernel_name = "test_kernel"
    kernel_string = """// This is where we define test_kernel
        #define SUM(A, B) (A + B)
        __kernel void test_kernel
        (char number, double factors, int * numbers, const unsigned long * moreNumbers) {
        numbers[get_global_id(0)] = SUM(numbers[get_global_id(0)] * factors[get_global_id(0)], number);
        }
        // /test_kernel
        """
    args = [np.byte(5), np.float64(4.6), np.int32([1, 2, 3]), np.uint64([3, 2, 111])]
    check_argument_list(kernel_name, kernel_string, args)
    # test that no exception is raised
    assert True


def test_check_argument_list7():
    kernel_name = "test_kernel"
    kernel_string = """#define SUM(A, B) (A + B)
        // In this file we define test_kernel
        __kernel void another_kernel (char number, double factors, int * numbers, const unsigned long * moreNumbers)
        __kernel void test_kernel
        (double number, double factors, int * numbers, const unsigned long * moreNumbers) {
        numbers[get_global_id(0)] = SUM(numbers[get_global_id(0)] * factors[get_global_id(0)], number);
        }
        // /test_kernel
        """
    args = [np.byte(5), np.float64(4.6), np.int32([1, 2, 3]), np.uint64([3, 2, 111])]
    assert_user_warning(check_argument_list, [kernel_name, kernel_string, args])


def test_check_tune_params_list():
    tune_params = dict(zip(["one_thing", "led_to_another", "and_before_you_know_it",
                            "grid_size_y"], [1, 2, 3, 4]))
    try:
        check_tune_params_list(tune_params)
        print("Expected a ValueError to be raised")
        assert False
    except ValueError as e:
        print(str(e))
        assert "Tune parameter grid_size_y with value 4 has a forbidden name!" == str(e)
    except Exception:
        print("Expected a ValueError to be raised")
        assert False


def test_check_tune_params_list2():
    tune_params = dict(zip(["rock", "paper", "scissors"], [1, 2, 3]))
    check_tune_params_list(tune_params)
    # test that no exception is raised
    assert True


def test_check_block_size_params_names_list():
    def test_warnings(function, args, number, warning_type):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            function(*args)
            # Verify some things
            assert len(w) == number
            for warn in w:
                assert issubclass(warn.category, warning_type)

    # check warning triggers for both unused blocksize names
    block_size_names = ["block_size_a", "block_size_b"]
    tune_params = dict(zip(["hyper", "ultra", "mega", "turbo"], [1, 2, 3, 4]))
    test_warnings(check_block_size_params_names_list, [block_size_names, tune_params], 2, UserWarning)

    # check warning does not triger when nondefault block size names are used correctly
    block_size_names = ["block_size_a", "block_size_b"]
    tune_params = dict(zip(["block_size_a", "block_size_b", "many_other_things"], [1, 2, 3]))
    test_warnings(check_block_size_params_names_list, [block_size_names, tune_params], 0, None)

    # check that a warning is issued when none of the default names are used and no alternative names are specified
    block_size_names = None
    tune_params = dict(zip(["block_size_a", "block_size_b", "many_other_things"], [1, 2, 3]))
    test_warnings(check_block_size_params_names_list, [block_size_names, tune_params], 1, UserWarning)

    # check that no error is raised when any of the default block size names is being used
    block_size_names = None
    tune_params = dict(zip(["block_size_x", "several_other_things"], [[1, 2, 3, 4], [2, 4]]))
    test_warnings(check_block_size_params_names_list, [block_size_names, tune_params], 0, None)


def test_get_kernel_string_func():
    # test whether passing a function instead of string works
    def gen_kernel(params):
        return "__global__ void kernel_name() { %s }" % params["block_size_x"]
    params = {"block_size_x": "//do that kernel thing!"}
    expected = "__global__ void kernel_name() { //do that kernel thing! }"
    answer = get_kernel_string(gen_kernel, params)
    assert answer == expected


def test_get_kernel_string_filename_not_found():
    # when the string looks like a filename, but the file does not exist
    # assume the string is not a filename after all
    bogus_filename = "filename_3456789.cu"
    answer = get_kernel_string(bogus_filename)
    assert answer == bogus_filename


def test_looks_like_a_filename1():
    string = "filename.c"
    assert looks_like_a_filename(string)


def test_looks_like_a_filename2():
    string = "__global__ void kernel_name() { //do that kernel thing! }"
    assert not looks_like_a_filename(string)


def test_read_write_file():
    filename = get_temp_filename()

    my_string = "this is the test string"
    try:
        write_file(filename, my_string)
        with open(filename, 'r') as f:
            answer = f.read()
        assert my_string == answer
        answer2 = read_file(filename)
        assert my_string == answer2

    finally:
        delete_temp_file(filename)


def test_normalize_verify_function():
    assert normalize_verify_function(None) is None

    def verify1(answer, result_host):
        return True
    v = normalize_verify_function(verify1)
    assert v(1, 2, atol=3)

    def verify2(answer, result_host, atol):
        return True
    v = normalize_verify_function(verify2)
    assert v(1, 2, atol=3)

    v = normalize_verify_function(lambda a, b: True)
    assert v(1, 2, atol=3)

    v = normalize_verify_function(lambda a, b, atol: True)
    assert v(1, 2, atol=3)


def test_process_cache():

    def assert_open_cachefile_is_correctly_parsed(cache):
        with open(cache, "r") as cachefile:
            filestr = cachefile.read()
            if filestr[-1] == ",":
                filestr = filestr[:-1]
            file_contents = filestr + "}\n}"
        cache_object = json.loads(file_contents)
        assert cache_object["device_name"] == "test_device"
        assert cache_object["kernel_name"] == "test_kernel"

    # get temp filename, but remove the file
    cache = get_temp_filename(suffix=".json")
    delete_temp_file(cache)

    kernel_options = Options(kernel_name="test_kernel")
    tuning_options = Options(cache=cache, tune_params=Options(x=[1, 2, 3, 4]), simulation_mode=False)
    runner = Options(dev=Options(name="test_device"), simulation_mode=False)

    try:
        # call process_cache without pre-existing cache
        process_cache(cache, kernel_options, tuning_options, runner)

        # check if file has been created
        assert os.path.isfile(cache)
        assert_open_cachefile_is_correctly_parsed(cache)
        assert tuning_options.cachefile == cache
        assert isinstance(tuning_options.cache, dict)
        assert len(tuning_options.cache) == 0

        # store one entry in the cache
        params = {"x": 4, "time": np.float32(0.1234)}
        store_cache("4", params, tuning_options)
        assert len(tuning_options.cache) == 1

        # close the cache
        close_cache(cache)

        # now test process cache with a pre-existing cache file
        process_cache(cache, kernel_options, tuning_options, runner)
        assert_open_cachefile_is_correctly_parsed(cache)

        assert tuning_options.cache["4"]["time"] == params["time"]

        # check that exceptions are raised when using a cache file for
        # a different kernel, device, or parameter set
        with pytest.raises(ValueError) as excp:
            kernel_options.kernel_name = "wrong_kernel"
            process_cache(cache, kernel_options, tuning_options, runner)
            assert "kernel" in str(excep.value)

        with pytest.raises(ValueError) as excp:
            runner.dev.name = "wrong_device"
            process_cache(cache, kernel_options, tuning_options, runner)
            assert "device" in str(excep.value)

        with pytest.raises(ValueError) as excp:
            tuning_options.tune_params["y"] = ["a", "b"]
            process_cache(cache, kernel_options, tuning_options, runner)
            assert "parameter" in str(excep.value)

    finally:
        delete_temp_file(cache)
        # pass


def test_process_metrics():
    params = {"x": 15, "b": 12}
    metrics = OrderedDict()
    metrics["y"] = lambda p: p["x"]

    # test if lambda function is correctly evaluated
    params = process_metrics(params, metrics)
    assert params["y"] == params["x"]

    # test if we can do the same with a string
    params = {"x": 15, "b": 12}
    metrics["y"] = "x"
    params = process_metrics(params, metrics)
    assert params["y"] == params["x"]

    # test if composability works correctly
    params = {"x": 15, "b": 12}
    metrics = OrderedDict()
    metrics["y"] = "x"
    metrics["z"] = "y"
    params = process_metrics(params, metrics)
    assert params["z"] == params["x"]

    # test ValueError is raised when metrics is not an OrderedDict
    with pytest.raises(ValueError):
        params = process_metrics(params, {})

    # test ValueError is raised when b already exists in params
    params = {"x": 15, "b": 12}
    metrics = OrderedDict()
    metrics["b"] = "x"
    with pytest.raises(ValueError):
        params = process_metrics(params, metrics)
