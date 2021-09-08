import os
import re
import json

from jsonschema import validate

from kernel_tuner import util

#specifies for a number of pre-defined objectives whether
#the objective should be minimized or maximized (boolean value denotes higher is better)
objective_default_map = {
    "time": False,
    "energy": False,
    "GFLOP/s": True,
    "TFLOP/s": True,
    "GB/s": True,
    "TB/s": True,
    "GFLOPS/W": True,
    "TFLOPS/W": True,
    "GFLOP/J": True,
    "TFLOP/J": True
}

def get_objective_defaults(objective, objective_higher_is_better):
    #use time as default objective and attempt to lookup objective_higher_is_better for known objectives
    objective = objective or "time"
    if objective_higher_is_better is None and objective in objective_default_map:
        objective_higher_is_better = objective_default_map[objective]
    else:
        raise ValueError(f"Please specify objective_higher_is_better for objective {objective}")
    return objective, objective_higher_is_better

schema_v1_0 = {
    "$schema": "https://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "version_number": {"type": "string"},
        "tunable_parameters": {"type": "array", "items": {"type": "string"}},
        "kernel_name": {"type": "string"},
        "kernel_string": {"type": "string"},
        "objective": {"type": "string"},
        "objective_higher_is_better": {"type": "boolean"},
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "device_name": {"type": "string"},
                    "problem_size": {"type": "string"}
                },
                "required": ["device_name", "problem_size", "tunable_parameters"]
            },
        },
    },
    "required": ["version_number", "tunable_parameters", "kernel_name", "objective", "data"]
}




class TuneResults():
    """ Object to represent the tuning results stored to file """

    def __init__(self, results_filename):
        #open results file
        if not os.path.isfile(results_filename):
            raise ValueError("Error: results_filename does not exist")
        meta, data = _read_results_file(results_filename)
        if len(data) < 1:
            raise ValueError("results file seems to be empty or did not load correctly")
        self.data = data
        self.meta = meta
        self.objective = meta["objective"]
        self.objective_higher_is_better = meta.get("objective_higher_is_better", False)

    def get_best_config(self, gpu_name="default", problem_size=None):
        """ get the best config based on these tuning results

            This function returns the overall best performing kernel configuration
            based on the tuning results for a given gpu_name and problem_size.

            If problem_size is not given this function will select a default configuration
            based on the tuning results for all problem_sizes and the given gpu_name.

            If gpu_name is not given this function will select a default configuration
            based on all tuning results.

            :param gpu_name: Name of the GPU for which the best configuration
                needs to be retrieved.
            :type gpu_name: string

            :param problem_size: The problem size for which the best configuration
                on the given gpu_name needs to be retrieved.
            :type problem_size: tuple, int, or string

            :returns: A dictionary with tunable parameters of the selected kernel
                kernel configuration.
            :rtype: dict
        """
        gpu_name = gpu_name.replace("-", "_").replace(" ", "_")

        if problem_size:
            if not isinstance(problem_size, str):
                if not isinstance(problem_size, (list, tuple)):
                    problem_size = (problem_size,)
                problem_size_str = "x".join(str(i) for i in problem_size)
            else:
                problem_size_str = problem_size

        gpu_match = [result for result in self.data if result["device_name"] == gpu_name]

        if gpu_match:
            gpu_ps_match = [result for result in gpu_match if problem_size and result["problem_size"] == problem_size_str]
            if gpu_ps_match:
                return _get_best_config_from_list(gpu_ps_match, self.objective, self.objective_higher_is_better)
            #problem size is not given or not among the results, so return a good default
            return _select_best_common_config(gpu_match, self.objective, self.objective_higher_is_better)

        #gpu is not among the results, so return a good default
        return _select_best_common_config(self.data, self.objective, self.objective_higher_is_better)


def store_results(results_filename, kernel_name, kernel_string, tune_params, problem_size, results, env, top=3, objective=None, objective_higher_is_better=None):
    """ stores tuning results to a JSON file

        Stores the top (3% by default) best kernel configurations in a JSON file.
        The results are stored for a specific device (retrieved by env.device_name)
        and for a specific problem_size. Any previous results stored in the file
        for this specific device and problem_size will be overwritten.

        :param results_filename: Filename of JSON file in which the results will be stored.
            Existing files may be overwritten.
        :type results_filename: string

        :param tune_params: The tunable parameters of this kernel.
        :type tune_params: dict

        :param problem_size: problem size used during tuning
        :type problem_size: tuple

        :param results: A list of dictionaries of all executed kernel configurations and their
            execution times, and possibly other user-defined metrics.
        :type results: list(dict)

        :param env: And a dictionary with information about the environment
            in which the tuning took place. This records device name, properties,
            version info, and so on. Typicaly this dictionary is returned by tune_kernel.
        :type env: dict

        :param top: Denotes the top percentage of results to store in the results file
        :type top: float

        :param objective: optimization objective to sort results on, consisting of a string that also
            occurs in results as a metric and a function, i.e. Python built-in functions min
            or max, that will be used to compare results.
        :type objective: tuple(string, callable)

    """

    objective, objective_higher_is_better = get_objective_defaults(objective, objective_higher_is_better)

    #filter results to only those that contain the objective
    results_filtered = [item for item in results if objective in item]

    #get top results
    if objective_higher_is_better:
        best_config = max(results_filtered, key=lambda x: x[objective])
    else:
        best_config = min(results_filtered, key=lambda x: x[objective])
    best = best_config[objective]
    top_range = top/100.0

    def top_result(item):
        current = item[objective]
        if objective_higher_is_better:
            return current > best * (1-top_range)
        return current < best * (1+top_range)
    top_results = [item for item in results_filtered if top_result(item)]

    #filter result items to just the tunable parameters and the objective
    filter_keys = list(tune_params.keys()) + [objective]
    top_results = [{k:item[k] for k in filter_keys} for item in top_results]

    #read existing results file
    if os.path.isfile(results_filename):
        meta, data = _read_results_file(results_filename)

        #validate consistency between arguments and results file
        if not kernel_name == meta["kernel_name"]:
            raise ValueError("Mismatch between given kernel_name and results file")
        if not all([param in meta["tunable_parameters"] for param in tune_params]):
            raise ValueError("Mismatch between tunable_parameters in results file and tune_params")
        if not objective == meta["objective"]:
            raise ValueError("Mismatch between given objective and results file")
    else:
        #new file
        meta = {}
        meta["version_number"] = "1.0"
        meta["kernel_name"] = kernel_name
        if kernel_string and not callable(kernel_string) and not isinstance(kernel_string, list):
            if util.looks_like_a_filename(kernel_string):
                meta["kernel_string"] = util.read_file(kernel_string)
            else:
                meta["kernel_string"] = kernel_string
        meta["objective"] = objective
        meta["objective_higher_is_better"] = objective_higher_is_better
        meta["tunable_parameters"] = list(tune_params.keys())
        data = []

    #insert new results into the list
    if not isinstance(problem_size, (list, tuple)):
        problem_size = (problem_size,)
    problem_size_str = "x".join(str(i) for i in problem_size)

    #replace all non alphanumeric characters with underscore
    dev_name = re.sub('[^0-9a-zA-Z]+', '_', env["device_name"].strip())

    #remove existing entries for this GPU and problem_size combination from the results if any
    data = [d for d in data if not (d["device_name"] == dev_name and d["problem_size"] == problem_size_str)]

    #extend the results with the top_results
    results = []
    for result in top_results:
        record = {"device_name": dev_name, "problem_size": problem_size_str, "tunable_parameters": {}}
        for k, v in result.items():
            if k in tune_params:
                record["tunable_parameters"][k] = v
        record[objective] = result[objective]
        results.append(record)
    data.extend(results)

    #write output file
    meta["data"] = data
    with open(results_filename, 'w') as fh:
        fh.write(json.dumps(meta, indent=""))


def create_device_targets(header_filename, results_filename, objective=None, objective_higher_is_better=None):
    """ create a header with device targets

        This function generates a header file with device targets for compiling
        a kernel with different parameters on different devices. The tuning
        results are stored in a JSON file created by store_results. Existing
        header_filename will be overwritten.

        This function only creates device targets and does not create problem_size
        specific targets. Instead it searches for configurations that perform well
        for different problem sizes and selects a single configuration to use
        for the kernel.

        The header file can be included in a kernel source file using:
        ``#include "header_filename.h"``

        The kernel can then be compiled for a specific device using:
        ``-DTARGET_GPU="name_of_gpu"``

        The header will also include a default value, which is chosen to perform well
        on different devices.

        :param header_filename: Filename of the to be created header file.
        :type header_filename: string

        :param results_filename: Filename of the JSON file that stores the tuning results.
        :type results_filename: string

        :param objective: optimization objective to sort results on, consisting of a string that also
            occurs in results as a metric and a function, i.e. Python built-in functions min
            or max, that will be used to compare results.
        :type objective: tuple(string, callable)
    """
    objective, objective_higher_is_better = get_objective_defaults(objective, objective_higher_is_better)

    #open results file
    results = TuneResults(results_filename)
    data = results.data

    #collect data for the if-block
    gpu_targets = list({r["device_name"] for r in data})
    targets = {}
    for gpu_name in gpu_targets:
        targets[gpu_name] = results.get_best_config(gpu_name)

    #select a good default from all good configs
    default_params = results.get_best_config()

    #write the header output file
    if_block = ""
    first = True
    for gpu_name, params in targets.items():
        if first:
            if_block += f"\n#ifdef TARGET_{gpu_name}\n"
            first = False
        else:
            if_block += f"\n#elif TARGET_{gpu_name}\n"
        if_block += "\n".join([f"#define {k} {v}" for k,v in params.items()])
        if_block += "\n"

    default_config = "\n".join([f"#define {k} {v}" for k,v in default_params.items()])

    template_header_file = f"""/* header file generated by Kernel Tuner, do not modify by hand */
#pragma once
#ifndef kernel_tuner /* only use these when not tuning */

{if_block}
#else /* default configuration */
{default_config}
#endif /* GPU TARGETS */

#endif /* kernel_tuner */
"""

    with open(header_filename, 'w') as fh:
        fh.write(template_header_file)




def _select_best_common_config(results, objective, objective_higher_is_better):
    """ return the most common config among results obtained on different problem sizes """
    results_table = {}
    total_performance = {}

    inverse_table = {}

    #for each configuration in the list
    for config in results:
        params = config["tunable_parameters"]

        config_str = util.get_instance_string(params)
        #count occurances
        results_table[config_str] = results_table.get(config_str,0) + 1
        #add to performance
        total_performance[config_str] = total_performance.get(config_str,0) + config[objective]
        #store mapping from config_str to the parameters
        inverse_table[config_str] = params

    #look for best config
    top_freq = max(results_table.values())
    best_configs = [k for k in results_table if results_table[k] == top_freq]

    #intersect total_performance with the best_configs
    total_performance = {k:total_performance[k] for k in total_performance if k in best_configs}

    #get the best config from this intersection
    if objective_higher_is_better:
        best_config_str = max(total_performance.keys(), key=lambda x: total_performance[x])
    else:
        best_config_str = min(total_performance.keys(), key=lambda x: total_performance[x])

    #lookup the tunable parameters of this configuration in the inverse table and return result
    return inverse_table[best_config_str]


def _get_best_config_from_list(configs, objective, objective_higher_is_better):
    """ return the tunable parameters of the best config from a list of configs """
    if objective_higher_is_better:
        best_config = max(configs, key=lambda x: x[objective])
    else:
        best_config = min(configs, key=lambda x: x[objective])
    best_config_params = {k:best_config[k] for k in best_config if k != objective}
    return best_config_params




def _read_results_file(results_filename):
    """ Reader for results file

        File format 1.0 specifies the following metadata
        "version_number": string e.g. "1.0"
        "tunable_parameters": list of strings
        "kernel_name": string
        "kernel_string": string with kernel code, optional
        "objective": string
        "objective_higher_is_better": True or False, default False
        "data": list of dicts
            each dict consists of the following keys:
            - "device_name": device name as reported by the device, with all non-alphanumeric characters replaced with "_"
            - "problem_size": a concatenated string of problem dimensions using "x" as separator
            - "tunable_parameters": a dict with all tunable parameters
            - "objective" as specified in the "objective" metadata

    """
    with open(results_filename, 'r') as fh:
        data = json.loads(fh.read())

    if "version_number" in data:
        if data["version_number"] == "1.0":
            return _parse_results_file_version_1_0(data)
        raise ValueError(f"Unknown results file version_number: {data['version_number']}")
    raise ValueError("Results fileformat not recognized")



def _parse_results_file_version_1_0(data):
    validate(instance=data, schema=schema_v1_0)

    meta_keys = ["kernel_name", "tunable_parameters", "objective", "version_number"]
    meta = {k: v for k, v in data.items() if k in meta_keys}
    meta["objective_higher_is_better"] = data.get("objective_higher_is_better", False)
    meta["kernel_string"] = data.get("kernel_string", "")
    entries = data["data"]

    #do some final checks against the metadata that cannot be handled by the JSON schema
    entry_keys = ["tunable_parameters"] + [meta["objective"]] + ["device_name", "problem_size"]
    for entry in entries:
        if not all([k in entry for k in entry_keys]):
            raise ValueError(f"Error while parsing results file, missing keys in: {entry}")
        if not all([k in entry["tunable_parameters"] for k in meta["tunable_parameters"]]):
            raise ValueError(f"Error while parsing results file, missing tunable parameter keys in: {entry}")

    return meta, entries
