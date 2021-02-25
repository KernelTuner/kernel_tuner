import os
import itertools
import json

import pytest

from kernel_tuner import integration
from kernel_tuner import util

@pytest.fixture()
def fake_results():
    #create fake results for testing
    tune_params = {"a": [1, 2, 4], "b": [4, 5, 6]}
    problem_size = 100
    parameter_space = itertools.product(*tune_params.values())
    results = [dict(zip(tune_params.keys(), element)) for element in parameter_space]
    for i,r in enumerate(results):
        r["time"] = 100.0+i
    env = {"device_name": "My GPU"}

    return "fake_kernel", "fake_string", tune_params, problem_size, parameter_space, results, env


def test_store_results(fake_results):

    filename = "temp_test_results_file.json"
    kernel_name, kernel_string, tune_params, problem_size, parameter_space, results, env = fake_results

    try:
        #test basic operation
        integration.store_results(filename, kernel_name, kernel_string, tune_params, problem_size, results, env, top=3)
        meta, stored_data = integration._read_results_file(filename)

        assert len([d for d in stored_data if d["device_name"] == "My_GPU" and d["problem_size"] == "100"]) == 3

        #test if results for a different problem_size values are added
        integration.store_results(filename, kernel_name, kernel_string, tune_params, 1000, results, env, top=3)
        meta, stored_data = integration._read_results_file(filename)

        assert len([d for d in stored_data if d["device_name"] == "My_GPU" and d["problem_size"] == "100"]) == 3
        assert len([d for d in stored_data if d["device_name"] == "My_GPU" and d["problem_size"] == "1000"]) == 3

        #test if results for a different GPU can be added
        integration.store_results(filename, kernel_name, kernel_string, tune_params, problem_size, results, {"device_name": "Another GPU"}, top=3)
        meta, stored_data = integration._read_results_file(filename)

        assert len(set([d["device_name"] for d in stored_data])) == 2

        #test if overwriting results works
        for i,r in enumerate(results):
            r["time"] = 50.0+i
        integration.store_results(filename, kernel_name, kernel_string, tune_params, problem_size, results, env, top=0.1)
        meta, stored_data = integration._read_results_file(filename)

        my_gpu_100_data = [d for d in stored_data if d["device_name"] == "My_GPU" and d["problem_size"] == "100"]
        assert len(my_gpu_100_data) == 1
        assert my_gpu_100_data[0]["time"] < 100

    finally:
        util.delete_temp_file(filename)



def test_setup_device_targets(fake_results):

    results_filename = "temp_test_results_file.json"
    header_filename = "temp_test_header_file.h"
    kernel_name, kernel_string, tune_params, problem_size, parameter_space, results, env = fake_results

    try:
        integration.store_results(results_filename, kernel_name, kernel_string, tune_params, problem_size, results, env, top=3)
        #results file
        #{'My_GPU': {'100': [{'a': 1, 'b': 4, 'time': 100.0}, {'a': 1, 'b': 5, 'time': 101.0}, {'a': 1, 'b': 6, 'time': 102.0}]}}

        integration.create_device_targets(header_filename, results_filename)

        with open(header_filename, 'r') as fh:
            output_str = fh.read()

        assert "#ifdef TARGET_My_GPU" in output_str
        assert "#define a 1" in output_str
        assert "#define b 4" in output_str

        #test output when more then one problem size is used, and best configuration is different
        for i,e in enumerate(results):
            if e['a'] == 1 and e['b'] == 4:
                e['time'] += 100
        integration.store_results(results_filename, kernel_name, kernel_string, tune_params, 1000, results, env, top=3)
        integration.create_device_targets(header_filename, results_filename, objective="time")

        with open(header_filename, 'r') as fh:
            output_str = fh.read()
        expected = "\n".join(["TARGET_My_GPU", "#define a 1", "#define b 5"])
        assert expected in output_str

        #test output when more then one problem size is used, and best configuration depends on total time
        for i,e in enumerate(results):
            if e['a'] == 1 and e['b'] == 6:
                e['time'] -= 3
        integration.store_results(results_filename, kernel_name, kernel_string, tune_params, 1000, results, env, top=3)
        integration.create_device_targets(header_filename, results_filename, objective="time")

        with open(header_filename, 'r') as fh:
            output_str = fh.read()
        expected = "\n".join(["TARGET_My_GPU", "#define a 1", "#define b 6"])
        assert expected in output_str

        #test output when more then one GPU is used
        for i,e in enumerate(results):
            if e['a'] == 1 and e['b'] == 6:
                e['time'] += 3.1
        env['device_name'] = "My_GPU2"
        integration.store_results(results_filename, kernel_name, kernel_string, tune_params, 1000, results, env, top=3)
        integration.create_device_targets(header_filename, results_filename, objective="time")

        with open(header_filename, 'r') as fh:
            output_str = fh.read()
        expected = "\n".join(["TARGET_My_GPU", "#define a 1", "#define b 6"])
        assert expected in output_str
        expected = "\n".join(["TARGET_My_GPU2", "#define a 1", "#define b 5"])
        assert expected in output_str
        expected = "\n".join(["#else /* default configuration */", "#define a 1", "#define b 5"])
        assert expected in output_str

    finally:
        util.delete_temp_file(results_filename)
        util.delete_temp_file(header_filename)


def test_setup_device_targets_max(fake_results):

    results_filename = "temp_test_results_file.json"
    header_filename = "temp_test_header_file.h"
    kernel_name, kernel_string, tune_params, problem_size, parameter_space, results, env = fake_results

    #add GFLOP/s as metric
    for i,e in enumerate(results):
        e['GFLOP/s'] = 1e5 / e['time']

    try:
        integration.store_results(results_filename, kernel_name, kernel_string, tune_params, problem_size, results, env, top=3, objective="GFLOP/s")
        integration.create_device_targets(header_filename, results_filename, objective="GFLOP/s")

        with open(header_filename, 'r') as fh:
            output_str = fh.read()
        assert "TARGET_My_GPU" in output_str
        assert "#define a 1" in output_str
        assert "#define b 4" in output_str

        #test output when more then one problem size is used, and best configuration is different
        for i,e in enumerate(results):
            if e['a'] == 1 and e['b'] == 4:
                e['time'] += 100
                e['GFLOP/s'] = 1e5 / e['time']
        integration.store_results(results_filename, kernel_name, kernel_string, tune_params, 1000, results, env, top=3, objective="GFLOP/s")
        integration.create_device_targets(header_filename, results_filename, objective="GFLOP/s")

        with open(header_filename, 'r') as fh:
            output_str = fh.read()
        expected = "\n".join(["TARGET_My_GPU", "#define a 1", "#define b 5"])
        assert expected in output_str


    finally:
        util.delete_temp_file(results_filename)
        util.delete_temp_file(header_filename)

