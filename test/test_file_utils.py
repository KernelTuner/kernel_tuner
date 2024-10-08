import json
from pathlib import Path

import pytest
from jsonschema import validate

from kernel_tuner.file_utils import get_input_file, output_file_schema, store_metadata_file, store_output_file
from kernel_tuner.util import delete_temp_file

from .test_runners import cache_filename, env, tune_kernel  # noqa: F401


def test_get_input_file(env):
    filename = Path(__file__).parent / "test_T1_input.json"
    assert filename.exists()
    contents = get_input_file(filename)
    assert isinstance(contents, dict)

def test_store_output_file(env):
    # setup variables
    filename = "test_output_file.json"

    try:
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

    finally:
        # clean up
        delete_temp_file(filename)


def test_store_metadata_file():
    # setup variables
    filename = "test_metadata_file.json"

    try:
        # run store_metadata_file
        try:
            store_metadata_file(filename)
        except FileNotFoundError:
            pytest.skip("'lshw' not present on this system")

        # retrieve metadata file
        _, schema = output_file_schema("metadata")
        with open(filename) as json_file:
            metadata_json = json.load(json_file)

        # validate
        validate(metadata_json, schema=schema)

    finally:
        # clean up
        delete_temp_file(filename)
