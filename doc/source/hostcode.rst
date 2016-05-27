

Tuning Host Code
================

With the kernel tuner it is also possible to tune the host code of your GPU programs, or even just any C function for that matter.
Tuning the host code can be useful when it contains parameters that have impact on the performance of kernel on the GPU, such as the number of
streams you use to execute a kernel across multiple streams. Another example is when you want to include the data transfers between
host and device into your tuning setup, or tune for different methods of moving data between host and device.

There are few differences with tuning just a single CUDA or OpenCL kernel, to list them:
* You have to specify the lang="C" option.
* The C function should return a ``float``
* You have to do your own timing in C

You have to specify the language as "C" because the kernel tuner will be calling a host function, this means that the kernel
tuner will have to interface with C and in fact uses a different backend. This also means you can use this way of tuning
without having PyCuda installed, because the C functions interface calls the CUDA compiler directly.

The C function should return a float, this is the convention used by the kernel tuner. The returned float is also the number
that you are tuning for. Meaning that this does not necessarily needs to be time, you could also optimize a program for
a different quality, as long as you can express that quality in a single floating-point value. When benchmarking an instance
of the parameter space the returned floats will be averaged for the multiple runs in the same way as with direct CUDA or OpenCL kernel tuning.

By itself the C language does not provide any very precise timing functions. If you are tuning the host code of a CUDA program you can use
CUDA Events to do the timing for you. However, if you are using plain C then you have to supply your own timing function. In the ``examples/c``
directory we have included a file ``timer.h`` that contains a very simple, but accurate timing function for Intel processors. To see how
to use that you can look at the minimal C example in ``examples/c/vector_add.py``. 

Tuning the number of streams
----------------------------

The following describes the example in ``examples/cuda/convolution_streams.py``.
In this example, the same convolution kernel is used as with correctness checking and convolution application example.

What is different is that we also supply the host code, which you can find in ``examples/cuda/convolution_streams.cu``. It is a bit
too long and complex to include here, but we will explain what it does. The idea behind the host code is that the kernel computation
is spread across a number of CUDA streams. In this way, it is possible to overlap the data transfers from host to device with kernel execution, and with
transfers from the device back to the host.

The way we divide the computation across streams is by dividing the problem in the y-dimension into chunks, where the first chunk is larger to account for the 
overlapping border between the data need by different streams. Before a kernel in stream `n` can start executing it is important that the data transfers in 
streams `n` and `n-1` has finished. To ensure the latter we use CUDA Event and cudaStreamWaitEvent, which halts stream `n` until the transfer in stream `n-1` has 
finished.

The way you use the kernel tuner to tune this CUDA program is very similar to when you are tuning only a single kernel, as you can see below:

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
* The source file contains host code rather kernel code
* "num_streams" is added to the tuning parameters
* "num_streams" is added to the "grid_div_y" list
* lang="C" is passed to tell "convolution_streams" is a C function
* ``filter`` is not passed as a constant memory argument, as this is done by the host code

Most differences have been explained, but we clarify a few things below.

The function that we are tuning is a C function that launches the CUDA kernel by itself, yet we supply the grid_div_x and 
grid_div_y lists. We are, however, not required to do so. The C function could just compute the grid dimensions in whatever way it sees fit. Using grid_div_y 
and grid_div_x at this point is matter of convience, and to support this the values grid_size_x and grid_size_y are inserted by the kernel tuner into the 
compiled C code. This way if you don't want to compute the grid size again in C you can just use the grid size as computed by the kernel tuner.

The filter is not passed separately as a constant memory argument, because the MemcpyToSymbol is now performed by the C host function itself. Since the code
is compiled differently, we have no direct reference to the module uploaded to the device and can therefore not perform this operation from Python. If you are
tuning host code, you have to perform all memory allocations, frees, and memcpy operations inside the C host code, that's the point of host code after all.





