.. highlight:: python
    :linenothreshold: 5



Correctness Verification
------------------------

Whenever you optimize a program for performance it is very important to
ensure that the program is still producing the correct output. What good
is a program that is fast but not correct?

Therefore an important feature of the kernel tuner is to verify the output
of every kernel instance in the parameter space. To use the kernel tuner
with correctness checking you need to pass the ``answer`` option to
``tune_kernel()``. Answer is a list that should match the order and types of
the kernel arguments. However, if an argument to the kernel is input-only
you may insert ``None`` at that location in the list.

After kernel compilation, but before benchmarking the kernel, the kernel
tuner runs the kernel once to verify the output it produces. For each
argument in the ``answer`` list that is not None, it will check the results
produced by the current kernel against the expected result specified in
``answer``. The comparison is currently implemented using numpy.allclose()
with an maximum allowed absolute error of 1e-6. If you want to use a 
difference tolerance value, use the optional argument ``atol``.

The example in ``examples/cuda/convolution_correct.py`` demonstrates how
to use the ``answer`` option of ``tune_kernel()``:

.. code-block:: python

    import numpy
    import kernel_tuner

    with open('convolution.cu', 'r') as f:
        kernel_string = f.read()

    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    input_size = ((problem_size[0]+16) * (problem_size[1]+16))

    output = numpy.zeros(size).astype(numpy.float32)
    input = numpy.random.randn(input_size).astype(numpy.float32)

    filter = numpy.random.randn(17*17).astype(numpy.float32)
    cmem_args= {'d_filter': filter }

    args = [output, input, filter]
    tune_params = dict()
    tune_params["block_size_x"] = [16*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    #compute the answer using a naive kernel
    params = { "block_size_x": 16, "block_size_y": 16 }
    results = kernel_tuner.run_kernel("convolution_naive", kernel_string,
        problem_size, args, params,
        grid_div_y=["block_size_y"], grid_div_x=["block_size_x"])

    #set non-output fields to None
    answer = [results[0], None, None]

    #start kernel tuning with correctness verification
    kernel_tuner.tune_kernel("convolution_kernel", kernel_string,
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x,
        verbose=True, cmem_args=cmem_args, answer=answer)

This example uses the ``run_kernel()`` function of the kernel tuner
to run a single kernel and return its results, with almost the same
interface as ``tune_kernel()``. In this example we run a naive CUDA
kernel whose results are trusted to be correct.

The ``answer`` list is constructed out of the results from the naive
kernel, but only includes the kernel arguments that are actually outputs.
The arguments that are input are replaced by a ``None`` value in the
``answer`` list before the list is passed to ``tune_kernel()``.

There are cases, however, where simply comparing the results computed on the device to precomputed values is not enough,
and more flexibility is necessary.
In this case, it is possible to use the ``verify`` option of ``tune_kernel()`` and specify a ``callable`` object that
implements a user-defined correctness check.
This function should accept three parameters: ``cpu_result``, ``gpu_result``, and ``atol``.
Although the name of the parameters can be different, their semantic is position dependent and reflected in the names
used in the documentation.

The example in ``examples/cuda/reduction.py`` demonstrates how to use the ``verify`` option of ``tune_kernel()``;
what follows is a snippet from the example:

.. code-block:: python

    # gpu_result
    args = [sum_x, x, n]
    # cpu_result
    reference = [numpy.sum(x), None, None]
    # custom verify function
    def verify_partial_reduce(cpu_result, gpu_result, atol=None):
        return numpy.isclose(cpu_result, numpy.sum(gpu_result), atol=atol)
    # call to tune_kernel()
    first_kernel, _ = tune_kernel("sum_floats", kernel_string, problem_size,
        args, tune_params, grid_div_x=[], verbose=True, answer=reference, verify=verify_partial_reduce)

The first argument, ``cpu_result``, is mapped to the NumPy array provided to the ``answer`` option; in this example it
is mapped to ``reference``.
The second argument, ``gpu_result``, is mapped to the NumPy array provided to the ``arguments`` option of
``tune_kernel()``; in this example it is mapped to ``args``.
The third argument, ``atol``, is set to ``None``; the default maximum allowed absolute error of 1e-6 is then used.

In the example, the user-defined ``verify`` function is used to compare the partial results, computed on the GPU,
to the final result, computed on the CPU.
The same could not be achieved just by using the ``answer`` option, because the number of elements in ``args[0]`` does
not necessarily match the number of elements in ``reference[0]`` in this example.
