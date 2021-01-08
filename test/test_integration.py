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

    return tune_params, problem_size, parameter_space, results, env


def test_store_results(fake_results):

    filename = "temp_test_results_file.json"
    tune_params, problem_size, parameter_space, results, env = fake_results

    try:
        #test basic operation
        integration.store_results(filename, tune_params, problem_size, results, env, top=3)
        with open(filename, 'r') as fh:
            stored_data = json.loads(fh.read())

        assert len(stored_data["My-GPU"]["100"]) == 3

        #test if results for a different problem_size values are added
        integration.store_results(filename, tune_params, 1000, results, env, top=3)
        with open(filename, 'r') as fh:
            stored_data = json.loads(fh.read())

        assert len(stored_data["My-GPU"]["100"]) == 3
        assert len(stored_data["My-GPU"]["1000"]) == 3

        #test if results for a different GPU can be added
        integration.store_results(filename, tune_params, problem_size, results, {"device_name": "Another GPU"}, top=3)
        with open(filename, 'r') as fh:
            stored_data = json.loads(fh.read())

        assert len(stored_data.keys()) == 2

        #test if overwriting results works
        for i,r in enumerate(results):
            r["time"] = 50.0+i
        integration.store_results(filename, tune_params, problem_size, results, env, top=0.1)
        with open(filename, 'r') as fh:
            stored_data = json.loads(fh.read())

        assert len(stored_data["My-GPU"]["100"]) == 1
        assert stored_data["My-GPU"]["100"][0]["time"] < 100


    finally:
        util.delete_temp_file(filename)


def test_setup_device_targets():

    results_filename = "temp_test_results_file.json"
    header_filename = "temp_test_header_file.h"

    #create_device_targets(header_filename, results_filename, objective=("time", min))

