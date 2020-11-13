"""
Minimal example of vector_add using a Jinja2 template.
"""

from kernel_tuner import tune_kernel
from jinja2 import Environment, FileSystemLoader
import json
import numpy


def generate_code(tuning_parameters):
    template_loader = FileSystemLoader(".")
    template_environment = Environment(loader=template_loader)
    template = template_environment.get_template("vector_add_jinja.cu")
    if tuning_parameters["vector_size"] == 1:
        real_type = tuning_parameters["real_type"]
    else:
        real_type = str(tuning_parameters["real_type"]) + str(tuning_parameters["vector_size"])
    return template.render(real_type=real_type, vector_size=tuning_parameters["vector_size"], block_size_x=tuning_parameters["block_size_x"], tiling_x=tuning_parameters["tiling_x"])


def tune():
    size = 10000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    arguments = [a, b, c, n]
    control = [None, None, a + b, None]

    tuning_parameters = dict()
    tuning_parameters["real_type"] = ["float"]
    tuning_parameters["block_size_x"] = [32 * i for i in range(1, 33)]
    tuning_parameters["vector_size"] = [1, 2, 4]
    tuning_parameters["tiling_x"] = [i for i in range(1, 33)]

    result = tune_kernel("vector_add", generate_code, size, arguments, tuning_parameters, lang="CUDA", grid_div_x=["block_size_x * vector_size * tiling_x"], answer=control)

    with open("vector_add_jinja.json", 'w') as fp:
        json.dump(result, fp)

    return result


if __name__ == "__main__":
    tune()
