from kernel_tuner.file_utils import store_output_file, store_metadata_file, output_file_schema, validate
from .test_integration import fake_results
from .test_runners import env, cache_filename, tune_kernel
import pytest
import json
import os


def test_store_output_file(env):
    # setup variables
    filename = "test_output_file.json"
    results, _ = tune_kernel(*env, cache=cache_filename, simulation_mode=True)
    tune_params = env[-1]

    # run store_output_file
    store_output_file(filename, results, tune_params)

    # retrieve output file
    _, schema = output_file_schema("results")
    with open(filename) as json_file:
        output_json = json.load(json_file)

    # validate
    validate(output_json, schema=schema)

    # clean up
    os.remove(filename)


def test_store_metadata_file():
    # setup variables
    filename = "test_metadata_file.json"

    # run store_metadata_file
    try:
        store_metadata_file(filename, target="nvidia")
    except FileNotFoundError:
        pytest.skip("'lshw' or 'nvidia-smi' not present on this system")

    # retrieve metadata file
    _, schema = output_file_schema("metadata")
    with open(filename) as json_file:
        metadata_json = json.load(json_file)

    # validate
    validate(metadata_json, schema=schema)

    # clean up
    os.remove(filename)
