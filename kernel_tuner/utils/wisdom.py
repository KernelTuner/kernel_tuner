"""Read and write wisdom files, which store tuning results in the format of Kernel Launcher.

Kernel Launcher (https://github.com/KernelTuner/kernel_launcher) stores the tuning results of a kernel in a
wisdom file named ``<key>.wisdom``. The file is in JSON Lines format: the first line is a header that lists the
names of the tunable parameters, and every following line is a record with the values of the tunable parameters
(``config``), the ``problem_size``, the objective (``time``), and the ``environment`` in which the configuration
was benchmarked, which includes the name of the GPU (``device_name``).

Kernel Launcher expects the problem size to be a list of integers. The tuning keys of the autotune decorator can
also contain other values, such as data types, so each record also stores the complete tuning key in ``key``.
Kernel Launcher ignores this field.
"""

import json
import logging
import os
import tempfile

WISDOM_VERSION = "1.0"
WISDOM_OBJECTIVE = "time"


def wisdom_file(directory, name):
    """Return the path of the wisdom file for the kernel with the given name."""
    return os.path.join(directory, f"{name}.wisdom")


def normalize_key(key):
    """Convert a tuning key to the JSON representation that is stored in wisdom files."""
    return json.loads(json.dumps(key, default=str))


def problem_size_of_key(key):
    """Return the integers in a tuning key, which Kernel Launcher uses as the problem size."""
    sizes = []

    def collect(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            sizes.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(normalize_key(key))
    return sizes


def read_wisdom(filename, tunable_parameters):
    """Read the records of a wisdom file.

    :param filename: Path of the wisdom file.
    :type filename: str

    :param tunable_parameters: Names of the tunable parameters. Records are only returned if the file stores the
        same tunable parameters, otherwise the file belongs to a different version of the kernel.
    :type tunable_parameters: list(str)

    :returns: The records in the file, an empty list if the file does not exist or does not match.
    :rtype: list(dict)
    """
    if not os.path.isfile(filename):
        return []
    with open(filename) as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        return []
    header = json.loads(lines[0])
    if header.get("tunable_parameters") != list(tunable_parameters):
        logging.warning(
            f"Ignoring wisdom file {filename}, it stores the tunable parameters {header.get('tunable_parameters')} "
            f"instead of {list(tunable_parameters)}"
        )
        return []
    return [json.loads(line) for line in lines[1:]]


def best_wisdom_config(records, key, device_name, tunable_parameters):
    """Return the configuration with the lowest time for the tuning key on the device, or None.

    :returns: A dictionary with the values of the tunable parameters, or None if no record matches.
    :rtype: dict or None
    """
    key = normalize_key(key)
    matching = [
        record
        for record in records
        if record.get("key") == key and record.get("environment", {}).get("device_name") == device_name
    ]
    if not matching:
        return None
    best = min(matching, key=lambda record: record[WISDOM_OBJECTIVE])
    return dict(zip(tunable_parameters, best["config"]))


def write_wisdom(filename, name, tunable_parameters, key, device_name, results):
    """Store the tuning results for a tuning key in a wisdom file.

    Records for the same tuning key and device are replaced, other records are kept.

    :param name: The name of the kernel, stored as the key of the wisdom file.
    :type name: str

    :param results: The valid results of tune_kernel, these contain the tunable parameters and the time.
    :type results: list(dict)
    """
    directory = os.path.dirname(os.path.abspath(filename))
    os.makedirs(directory, exist_ok=True)
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        with open(filename) as handle:
            stored_parameters = json.loads(handle.readline()).get("tunable_parameters")
        if stored_parameters != list(tunable_parameters):
            # do not overwrite the results of another version of the kernel
            logging.warning(f"Not storing tuning results in {filename}, it stores other tunable parameters")
            return
    key = normalize_key(key)
    records = [
        record
        for record in read_wisdom(filename, tunable_parameters)
        if record.get("key") != key or record.get("environment", {}).get("device_name") != device_name
    ]
    problem_size = problem_size_of_key(key)
    for result in results:
        records.append(
            {
                "config": [result[param] for param in tunable_parameters],
                "problem_size": problem_size,
                WISDOM_OBJECTIVE: result[WISDOM_OBJECTIVE],
                "environment": {"device_name": device_name},
                "key": key,
            }
        )

    header = {
        "tunable_parameters": list(tunable_parameters),
        "version": WISDOM_VERSION,
        "objective": WISDOM_OBJECTIVE,
        "key": name,
    }
    lines = [json.dumps(header)] + [json.dumps(record, default=str) for record in records]
    # write to a temporary file first, so that a process that reads the file never sees a partial file
    with tempfile.NamedTemporaryFile("w", dir=directory, suffix=".tmp", delete=False) as handle:
        handle.write("\n".join(lines) + "\n")
        temp_name = handle.name
    os.replace(temp_name, filename)
