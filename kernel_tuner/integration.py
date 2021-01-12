import os
import json

from kernel_tuner import util


def store_results(results_filename, tune_params, problem_size, results, env, top=3, objective=("time", min)):
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

    #filter results to only those that contain the objective
    results_filtered = [item for item in results if objective[0] in item]

    #get top results
    best_config = objective[1](results_filtered, key=lambda x: x[objective[0]])
    best = best_config[objective[0]]
    top_range = top/100.0

    def top_result(item):
        current = item[objective[0]]
        if objective[1] == min:
            return current < best * (1+top_range)
        elif objective[1] == max:
            return current > best * (1-top_range)
        else:
            raise ValueError("only min or max are supported to compare results")

    top_results = [item for item in results_filtered if top_result(item)]

    #filter result items to just the tunable parameters and the objective
    filter_keys = list(tune_params.keys()) + [objective[0]]
    top_results = [{k:item[k] for k in filter_keys} for item in top_results]

    #read results file
    if os.path.isfile(results_filename):
        with open(results_filename, 'r') as fh:
            data = json.loads(fh.read())
    else:
        data = {}

    #insert new results into the database
    if not isinstance(problem_size, (list, tuple)):
        problem_size = (problem_size,)
    problem_size_str = "x".join(str(i) for i in problem_size)

    dev_name = env["device_name"].strip().replace(" ", '-')

    datum_insert = {problem_size_str: top_results}
    if dev_name in data:
        data[dev_name].update(datum_insert)
    else:
        data[dev_name] = datum_insert

    #write output file
    with open(results_filename, 'w') as fh:
        fh.write(json.dumps(data))


def create_device_targets(header_filename, results_filename, objective=("time", min)):
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

    #open results file
    if not os.path.isfile(results_filename):
        raise ValueError("Error: results_filename does not exist")
    with open(results_filename, 'r') as fh:
        data = json.loads(fh.read())

    if len(data) < 1:
        raise ValueError("results file seems to be empty or did not load correctly")

    #collect data for the if-block
    targets = {}
    for gpu_name, problem_sizes in data.items():
        #key is a string that denotes the GPU
        #value is a dict of problem_size keys and lists of good configs for each problem_size

        if len(problem_sizes) == 1:
            best_config = objective[1](list(problem_sizes.values())[0], key=lambda x: x[objective[0]])
            best_config_params = {k:best_config[k] for k in best_config.keys() if k != objective[0]}
        else:
            best_config_params = _select_best_common_config(problem_sizes, objective)
        targets[gpu_name] = best_config_params

    #select a good default from all good configs
    merge_results = {}
    for gpu, problem_sizes in data.items():
        for problem_size, configs in problem_sizes.items():
            merge_results[gpu + "_" + problem_size] = configs
    default_params = _select_best_common_config(merge_results, objective)

    #write the header output file
    if_block = ""
    for gpu_name, params in targets.items():
        if_block += f"\n#elif TARGET_GPU == {gpu_name}\n"
        if_block += "\n".join([f"#define {k} {v}" for k,v in params.items()])
        if_block += "\n"

    default_config = "\n".join([f"#define {k} {v}" for k,v in default_params.items()])

    template_header_file = f"""/* header file generated by Kernel Tuner, do not modify by hand */
#pragma once
#ifndef kernel_tuner /* only use these when not tuning */

#ifndef TARGET_GPU /* default configuration */
{default_config}
{if_block}
#endif /* TARGET_GPU */

#endif /* kernel_tuner */
"""

    with open(header_filename, 'w') as fh:
        fh.write(template_header_file)




def _select_best_common_config(results, objective):
    """ return the most common config among results obtained on different problem sizes """

    results_table = {}
    total_performance = {}

    inverse_table = {}

    #for each problem_size in the results dictionary
    for key, value in results.items():
        #for each configuration in the list
        for config in value:
            params = {k:config[k] for k in config.keys() if k != objective[0]}

            config_str = util.get_instance_string(params)
            #count occurances
            results_table[config_str] = results_table.get(config_str,0) + 1
            #add to performance
            total_performance[config_str] = total_performance.get(config_str,0) + config[objective[0]]
            #store mapping from config_str to the parameters
            inverse_table[config_str] = params

    #look for best config
    top_freq = max(results_table.values())
    best_configs = [k for k in results_table.keys() if results_table[k] == top_freq]

    #intersect total_performance with the best_configs
    total_performance = {k:total_performance[k] for k in total_performance.keys() if k in best_configs}

    #get the best config from this intersection
    best_config_str = objective[1](total_performance.keys(), key=lambda x: total_performance[x])

    #lookup the tunable parameters of this configuration in the inverse table and return result
    return inverse_table[best_config_str]
