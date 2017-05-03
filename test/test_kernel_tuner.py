from __future__ import print_function

import numpy
from nose.tools import raises

from .context import skip_if_no_cuda_device, skip_if_no_opencl

import kernel_tuner.cuda as cuda
import kernel_tuner.opencl as opencl
from kernel_tuner.util import *
from kernel_tuner.core import *


def test_get_grid_dimensions1():
    problem_size = (1024, 1024, 1)
    params = {"block_x": 41, "block_y": 37}

    grid_div = ( ["block_x"], ["block_y"], None )

    grid = get_grid_dimensions(problem_size, params,
                    grid_div)

    assert len(grid) == 3
    assert isinstance(grid[0], int)
    assert isinstance(grid[1], int)

    assert grid[0] == 25
    assert grid[1] == 28
    assert grid[2] == 1

    grid = get_grid_dimensions(problem_size, params,
                    (grid_div[0], None, None))

    assert grid[0] == 25
    assert grid[1] == 1024
    assert grid[2] == 1

    grid = get_grid_dimensions(problem_size, params,
                    (None, grid_div[1], None))

    assert grid[0] == 1024
    assert grid[1] == 28
    assert grid[2] == 1

def test_get_grid_dimensions2():
    problem_size = (1024, 1024, 1)
    params = {"block_x": 41, "block_y": 37}

    grid_div_x = ["block_x*8"]
    grid_div_y = ["(block_y+2)/8"]

    grid = get_grid_dimensions(problem_size, params,
                    (grid_div_x, grid_div_y, None))

    assert grid[0] == 4
    assert grid[1] == 256

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

@raises(TypeError)
def test_get_problem_size3():
    problem_size = (3.8, "num_blocks_y*3")
    params = {"num_blocks_y": 57}
    get_problem_size(problem_size, params)

def test_get_grid_dimensions5():
    problem_size = (1024, 1024, 1)
    params = {"block_x": 41, "block_y": 37}

    grid_div_x = ["block_x", "block_y"]
    grid_div_y = ["(block_y+2)/8"]

    grid = get_grid_dimensions(problem_size, params,
                    (grid_div_x, grid_div_y, None))

    assert grid[0] == 1
    assert grid[1] == 256
    assert grid[2] == 1


def test_get_grid_dimensions6():

    problem_size = (numpy.int32(1024), numpy.int64(1024), 1)
    params = {"block_x": 41, "block_y": 37}

    grid_div_x = ["block_x", "block_y"]
    grid_div_y = ["(block_y+2)/8"]

    grid = get_grid_dimensions(problem_size, params,
                    (grid_div_x, grid_div_y, None))

    assert grid[0] == 1
    assert grid[1] == 256
    assert grid[2] == 1



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

    output = prepare_kernel_string(kernel, params, (3,7))
    expected = "#define is 8\n#define grid_size_y 7\n#define grid_size_x 3\nthis is a weird kernel"
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
    restrictions = [["a==b+c"], ["a==b+c", "b==b", "a-b==c"], ["a==b+c", "b!=b", "a-b==c"]]
    expected = [True, True, False]
    #test the call returns expected
    for r,e in zip(restrictions, expected):
        answer = check_restrictions(r, params.values(), params.keys(), False)
        print(answer)
        assert answer == e

def test_detect_language1():
    lang = None
    kernel_string = "__global__ void vector_add( ... );"
    lang = detect_language(lang, kernel_string)
    assert lang == "CUDA"

def test_detect_language2():
    lang = None
    kernel_string = "__kernel void vector_add( ... );"
    lang = detect_language(lang, kernel_string)
    assert lang == "OpenCL"

def test_detect_language3():
    lang = None
    kernel_string = "blabla"
    lang = detect_language(lang, kernel_string)
    assert lang == "C"

def test_detect_language4():
    lang = "CUDA"
    kernel_string = "blabla"
    try:
        lang = detect_language(lang, kernel_string)
        assert lang == "CUDA"
    except Exception:
        assert False


def test_get_device_interface1():
    skip_if_no_cuda_device()
    lang = "CUDA"
    dev = get_device_interface(lang, 0, 0)
    assert isinstance(dev, cuda.CudaFunctions)

def test_get_device_interface2():
    skip_if_no_opencl()
    lang = "OpenCL"
    dev = get_device_interface(lang, 0, 0)
    assert isinstance(dev, opencl.OpenCLFunctions)

@raises(Exception)
def test_get_device_interface3():
    lang = "blabla"
    get_device_interface(lang, 0, 0)

def test_check_argument_list1():
    args = [numpy.int32(5), 'blah', numpy.array([1, 2, 3])]
    try:
        check_argument_list(args)
        print("Expected a TypeError to be raised")
        assert False
    except TypeError as e:
        print(str(e))
        assert "at position 1" in str(e)
    except Exception:
        print("Expected a TypeError to be raised")
        assert False

def test_check_argument_list2():
    args = [numpy.int32(5), numpy.float64(4.6), numpy.array([1, 2, 3])]
    check_argument_list(args)
    #test that no exception is raised
    assert True


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
