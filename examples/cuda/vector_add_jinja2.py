"""
More advanced example of vector_add using a Jinja2 template.
"""
import json

import numpy
from jinja2 import Template

from kernel_tuner import tune_kernel


def tune():

    with open('vector_add_jinja.cu', 'r') as template_file:
        template = Template(template_file.read())

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

    result = tune_kernel("vector_add", template.render, size, arguments, tuning_parameters, lang="CUDA", grid_div_x=["block_size_x * vector_size * tiling_x"], answer=control)

    with open("vector_add_jinja.json", 'w') as fp:
        json.dump(result, fp)

    return result


if __name__ == "__main__":
    tune()
