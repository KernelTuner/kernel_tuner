import os
import itertools
import json

from kernel_tuner import integration
from kernel_tuner import util

def test_store_results():

    filename = "temp_test_results_file.json"
    tune_params = {"a": [1, 2, 4], "b": [4, 5, 6]}
    problem_size = 100

    #create fake results for testing
    parameter_space = itertools.product(*tune_params.values())
    results = [dict(zip(tune_params.keys(), element)) for element in parameter_space]
    for i,r in enumerate(results):
        r["time"] = 100.0+i
    env = {"device_name": "My GPU"}

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
