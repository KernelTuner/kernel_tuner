.. highlight:: python
    :linenothreshold: 5



Tuning Host Code
----------------

With the Kernel Tuner it is also possible to tune the host code of your GPU programs, or even just any C function for that matter.
Tuning host code can be useful when it contains parameters that have impact on the performance of kernel on the GPU, such as the number of
streams to use when executing a kernel across multiple streams. Another example is when you want to include the data transfers between
host and device into your tuning setup, or tune for different methods of moving data between host and device.

There are few differences with tuning just a single CUDA or OpenCL kernel, to list them:  
 * You have to specify the lang="C" option
 * The C function should return a ``float``
 * You have to do your own timing and error handling in C
 * Data is not automatically copied to and from device memory. To use an array in host memory, pass in a :mod:`numpy` array. To use an array
   in device memory, pass in a :mod:`cupy` array.

You have to specify the language as "C" because the Kernel Tuner will be calling a host function. This means that the Kernel
Tuner will have to interface with C and in fact uses a different backend. This also means you can use this way of tuning
without having PyCuda installed, because the C functions interface calls the CUDA compiler directly.

The C function should return a float, this is the convention used by the Kernel Tuner. The returned float is also the number
that you are tuning for. Meaning that this does not necessarily needs to be time, you could also optimize a program for
a different quality, as long as you can express that quality in a single floating-point value. When benchmarking an instance
of the parameter space the returned floats will be averaged for the multiple runs in the same way as with direct CUDA or OpenCL kernel tuning.

By itself the C language does not provide any very precise timing functions. If you are tuning the host code of a CUDA program you can use
CUDA Events to do the timing for you. However, if you are using plain C then you have to supply your own timing function.
In the `C vector add example <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/c/vector_add.py>`__ we are using the ``omp_get_wtime()`` function from OpenMP to measure time on the CPU.

Tuning the number of streams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following describes the example in ``examples/cuda/convolution_streams.py``.
In this example, the same convolution kernel is used as with correctness checking and convolution application example.

What is different is that we also supply the host code, which you can find in ``examples/cuda/convolution_streams.cu``. It is a bit
too long and complex to include here, but we will explain what it does. The idea behind the host code is that the kernel computation
is spread across a number of CUDA streams. In this way, it is possible to overlap the data transfers from host to device with kernel execution, and with
transfers from the device back to the host.

The way we split the computation across streams is by dividing the problem in the y-dimension into chunks. The data transferred by the first stream is slightly 
larger to account for the overlapping border between the data needed by different streams. Before the kernel in stream `n` can start executing the data transfers 
in streams `n` and `n-1` have to be finished. To ensure the latter, we use CUDA Events and in particular cudaStreamWaitEvent(), which halts stream `n` until the 
transfer in stream `n-1` has finished.

The way you use the Kernel Tuner to tune this CUDA program is very similar to when you are tuning a CUDA kernel directly, as you can see below:

.. code-block:: python

    with open('convolution_streams.cu', 'r') as f:
        kernel_string = f.read()

    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    input_size = (problem_size[0]+16) * (problem_size[0]+16)

    output = numpy.zeros(size).astype(numpy.float32, order='C')
    input = numpy.random.randn(input_size).astype(numpy.float32, order='C')
    filter = numpy.random.randn(17*17).astype(numpy.float32)
    args = [output, input, filter]

    tune_params = dict()
    tune_params["block_size_x"] = [16*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

    tune_params["num_streams"] = [2**i for i in range(6)]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y", "num_streams"]

    kernel_tuner.tune_kernel("convolution_streams", kernel_string,
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x, verbose=True, lang="C")

In fact, the only differences with the simple convolution example are:  
 * The source file also contains host code 
 * "num_streams" is added to the tuning parameters
 * "num_streams" is added to the "grid_div_y" list
 * The kernel_name "convolution_streams" is a C function
 * lang="C" is used to tell this is a C function
 * ``filter`` is not passed as a constant memory argument

Most differences have been explained, but we clarify a few things below.

The function that we are tuning is a C function that launches the CUDA kernel by itself, yet we supply the grid_div_x and 
grid_div_y lists. We are, however, not required to do so. The C function could just compute the grid dimensions in whatever way it sees fit. Using grid_div_y 
and grid_div_x at this point is matter of choice. To support this convenience, the values grid_size_x and grid_size_y are inserted by the Kernel Tuner into the 
compiled C code. This way, you don't have to compute the grid size in C, you can just use the grid dimensions as computed by the Kernel Tuner.

The filter is not passed separately as a constant memory argument, because the CudaMemcpyToSymbol operation is now performed by the C host function. Also, 
because the code is compiled differently, we have no direct reference to the compiled module that is uploaded to the device and therefore we can not perform this 
operation directly from Python. If you are tuning host code, you have the option to perform all memory allocations, frees, and memcpy operations inside the C host code, 
that's the purpose of host code after all. That is also why you have to do the timing yourself in C, as you may not want to include the time spent on memory 
allocations and other setup into your time measurements.





