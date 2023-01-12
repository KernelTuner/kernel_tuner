import os
import json
import subprocess
import xmltodict

from importlib.metadata import requires, version, PackageNotFoundError
from packaging.requirements import Requirement

from jsonschema import validate

from kernel_tuner import util

schema_dir = os.path.dirname(os.path.realpath(__file__)) + "/schema"


def output_file_schema(target):
    """ Get the requested JSON schema and the version number

    :param target: Name of the T4 schema to return, should be any of ['output', 'metadata']
    :type target: string

    :returns: the current version of the T4 schemas and the JSON string of the target schema
    :rtype: string, string

    """
    current_version = "1.0.0"
    output_file = schema_dir + f"/T4/{current_version}/{target}-schema.json"
    with open(output_file, 'r') as fh:
        json_string = json.load(fh)
    return current_version, json_string


def get_configuration_validity(objective) -> str:
    """ Convert internal Kernel Tuner error to string """
    if not isinstance(objective, util.ErrorConfig):
        return "correct"
    else:
        if isinstance(objective, util.CompilationFailedConfig):
            return "compile"
        elif isinstance(objective, util.RuntimeFailedConfig):
            return "runtime"
        else:
            return "constraints"


def store_output_file(output_filename, results, tune_params, objective="time"):
    """ Store the obtained auto-tuning results in a JSON output file

    This function produces a JSON file that adheres to the T4 auto-tuning output JSON schema.

    :param output_filename: Name of the to be created output file
    :type output_filename: string

    :param results: Results list as return by tune_kernel
    :type results: list of dicts

    :param tune_params: Tunable parameters as passed to tune_kernel
    :type tune_params: OrderedDict

    :param objective: The objective used during auto-tuning, default is 'time'.
    :type objective: string

    """
    if output_filename[-5:] != ".json":
        output_filename += ".json"

    timing_keys = [
        "compile_time", "benchmark_time", "framework_time", "strategy_time",
        "verification_time"
    ]
    not_measurement_keys = list(
        tune_params.keys()) + timing_keys + ["timestamp"] + ["times"]

    output_data = []

    for result in results:

        out = {}

        out["timestamp"] = result["timestamp"]
        out["configuration"] = {
            k: v
            for k, v in result.items() if k in tune_params
        }

        # collect configuration specific timings
        timings = dict()
        timings["compilation"] = result["compile_time"]
        timings["benchmark"] = result["benchmark_time"]
        timings["framework"] = result["framework_time"]
        timings["search_algorithm"] = result["strategy_time"]
        timings["validation"] = result["verification_time"]
        timings["runtimes"] = result["times"]
        out["times"] = timings

        # encode the validity of the configuration
        out["invalidity"] = get_configuration_validity(result[objective])

        # Kernel Tuner does not support producing results of configs that fail the correctness check
        # therefore correctness is always 1
        out["correctness"] = 1

        # measurements gathers everything that was measured
        measurements = []
        for key, value in result.items():
            if key not in not_measurement_keys:
                measurements.append(
                    dict(name=key,
                         value=value,
                         unit="ms" if key.startswith("time") else ""))
        out["measurements"] = measurements

        # objectives
        # In Kernel Tuner we currently support only one objective at a time, this can be a user-defined
        # metric that combines scores from multiple different quantities into a single value to support
        # multi-objective tuning however.
        out["objectives"] = [objective]

        # append to output
        output_data.append(out)

    # write output_data to a JSON file
    version, _ = output_file_schema("results")
    output_json = dict(results=output_data, schema_version=version)
    with open(output_filename, 'w+') as fh:
        json.dump(output_json, fh)


def get_dependencies(package='kernel_tuner'):
    """ Get the Python dependencies of Kernel Tuner currently installed and their version numbers """
    requirements = requires(package)
    deps = [Requirement(req).name for req in requirements]
    depends = []
    for dep in deps:
        try:
            depends.append(f"{dep}=={version(dep)}")
        except PackageNotFoundError:
            # uninstalled packages can not have been used to produce these results
            # so it is safe to ignore
            pass
    return depends


def get_device_query(target):
    """ Get the information about GPUs in the current system, target is any of ['nvidia', 'amd'] """
    if target == "nvidia":
        nvidia_smi_out = subprocess.run(["nvidia-smi", "--query", "-x"],
                                        capture_output=True)
        nvidia_smi = xmltodict.parse(nvidia_smi_out.stdout)
        del nvidia_smi["nvidia_smi_log"]["gpu"]["processes"]
        return nvidia_smi
    elif target == "amd":
        rocm_smi_out = subprocess.run(["rocm-smi", "--showallinfo", "--json"],
                                      capture_output=True)
        return json.loads(rocm_smi_out.stdout)
    else:
        raise ValueError("get_device_query target not supported")


def store_metadata_file(metadata_filename, target="nvidia"):
    """ Store the metadata about the current hardware and software environment in a JSON output file

    This function produces a JSON file that adheres to the T4 auto-tuning metadata JSON schema.

    :param metadata_filename: Name of the to be created metadata file
    :type metadata_filename: string

    :param target: Target specifies whether to include the metadata of the 'nvidia' or 'amd' GPUs in the system
    :type target: string

    """
    if metadata_filename[-5:] != ".json":
        metadata_filename += ".json"
    metadata = {}

    # lshw only works on Linux, this intentionally raises a FileNotFoundError when ran on systems that do not have it
    lshw_out = subprocess.run(["lshw", "-json"], capture_output=True)
    metadata["hardware"] = dict(lshw=json.loads(lshw_out.stdout))

    # only works if nvidia-smi (for NVIDIA) or rocm-smi (for AMD) is present, raises FileNotFoundError when not present
    device_query = get_device_query(target)

    metadata["environment"] = dict(device_query=device_query,
                                   requirements=get_dependencies())

    # write metadata to JSON file
    version, _ = output_file_schema("metadata")
    metadata_json = dict(metadata=metadata, schema_version=version)
    with open(metadata_filename, 'w+') as fh:
        json.dump(metadata_json, fh, indent="  ")
