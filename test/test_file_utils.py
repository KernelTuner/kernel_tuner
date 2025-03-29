import json

import pytest
import ctypes
from jsonschema import validate
import numpy as np
import warnings
try:
    from hip import hip
except:
    hip = None

from kernel_tuner.file_utils import output_file_schema, store_metadata_file, store_output_file
from kernel_tuner.util import delete_temp_file, check_argument_list
from .context import skip_if_no_hip

from .test_runners import cache_filename, env, tune_kernel  # noqa: F401


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

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

@skip_if_no_hip
def test_check_argument_list_device_array():
    """Test check_argument_list with DeviceArray"""
    float_kernel = """
    __global__ void simple_kernel(float* input) {
        // kernel code
    }
    """
    host_array = np.ones((100,), dtype=np.float32)
    num_bytes = host_array.size * host_array.itemsize
    device_array = hip_check(hip.hipMalloc(num_bytes))
    device_array.configure(
        typestr="float32",
        shape=host_array.shape,
        itemsize=host_array.itemsize
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_argument_list("simple_kernel", float_kernel, [device_array])
