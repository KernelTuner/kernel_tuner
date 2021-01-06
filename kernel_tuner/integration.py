import os
import json


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
