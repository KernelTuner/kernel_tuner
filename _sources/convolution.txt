.. highlight:: python
    :linenothreshold: 5


Convolution Example
-------------------

2D Convolution is widely used in image processing for many purposes
including filtering. A naive CUDA kernel for 2D Convolution would be:

.. code-block:: c

    __global__ void convolution_kernel(float *output, float *input, float *filter) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i, j;
        float sum = 0.0;

        if (y < image_height && x < image_width) {

            for (j = 0; j < filter_height; j++) {
                for (i = 0; i < filter_width; i++) {
                    sum += input[(y + j) * input_width + (x + i)] * filter[j * filter_width + i];
                }
            }

            output[y * image_width + x] = sum;
        }
    }

The idea is that this kernel is launched with a CUDA thread for 
each pixel in the output image. Note that to avoid confusion 
around the term kernel, we refer to the convolution filter as a 
filter.

Setup tuning parameters
~~~~~~~~~~~~~~~~~~~~~~~

Say we are unaware of which combination of thread block 
dimensions gives the best performance on a given GPU. We can use 
the kernel_tuner's ``tune_kernel()`` function to find the best 
performing kernel configuration.

The above kernel uses built-in variables ``blockDim.x`` and 
``blockDim.y``. However, if we replace these with ``block_size_x`` and 
``block_size_y``, the kernel_tuner will replace them with a 
different combination of values every time it compiles the 
kernel.

We have to tell the kernel_tuner what values it should try for 
``block_size_x`` and ``block_size_y``. Therefore, we create a 
dictionary called tune_params and store a number of possible 
values for ``block_size_x`` and ``block_size_y`` that seem 
reasonable.

::

    tune_params = dict()
    tune_params["block_size_x"] = [16*i for i in range(1,9)] #[16, 32, 48, 64, 80, 96, 112, 128]
    tune_params["block_size_y"] = [2**i for i in range(6)]   #[1, 2, 4, 8, 16, 32]
   
Let's say we also have two other parameters (not shown in the
kernel code above): ``tile_size_x`` and ``tile_size_y``,
which increase the amount of work per thread block by a factor of
``tile_size_x`` in the x-direction and by a factor of
``tile_size_y`` in the
y-direction. We add these to the tuning parameters:

::

    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

The kernel tuner will try every possible combination of tuning 
parameters that we supply to it, so far that's already:
``8 * 6 * 3 * 3 = 432`` different combinations!

Setup kernel arguments
~~~~~~~~~~~~~~~~~~~~~~

We also have to tell ``tune_kernel()`` how it is supposed to call 
our kernel. To this end, we create a list of Numpy objects that 
we call args. Arrays can be passed as numpy.ndarray objects. 
Single values can be passed by value, for example as numpy.int32 
or numpy.float32.

:: 

    output = numpy.zeros(size).astype(numpy.float32)
    input = numpy.random.randn(input_size).astype(numpy.float32)
    filter = numpy.random.randn(17*17).astype(numpy.float32)
    args = [output, input, filter]

Note that the order within args should match the order of 
arguments to the kernel. Also be sure that the types correspond
with the types expected by the CUDA kernel.

Setup grid dimensions
~~~~~~~~~~~~~~~~~~~~~

Remember that the kernel originally used one thread per pixel 
in the output image. We have to tell ``tune_kernel()`` how the 
grid dimensions are computed. Therefore we set the following:

::

    problem_size = (width, height)    #output image width/height 
    grid_div_x = ["block_size_x"]
    grid_div_y = ["block_size_y"]

However, since we have also introduced ``tile_size_x`` and 
``tile_size_y`` that increase the amount of work per thread 
block, we have to decrease the number of thread blocks to be 
created even further.

::

    problem_size = (width, height)    #output image width/height 
    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

The above code tells the kernel tuner to compute the grid x-dimension
by taking the problem size in the x-direction and divide it by the value of
each tuning parameter in the grid_div_x list. The same holds for the
grid y-dimension.

The kernel tuner currently assumes that the thread block dimensions
are specified through the values of "block_size_x", "block_size_y",
and "block_size_z" in the tuning parameters. If one or more of
these values are not among the tuning parameters it will assume
256, 1, and 1, as thread block dimensions, respectively.

Putting it all together
~~~~~~~~~~~~~~~~~~~~~~~

Now putting it all together, we finally get to call ``tune_kernel()``:

::

    import numpy
    import kernel_tuner

    with open('convolution.cu', 'r') as f:
        kernel_string = f.read()

    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    input_size = (problem_size[0]+16) * (problem_size[0]+16)

    output = numpy.zeros(size).astype(numpy.float32)
    input = numpy.random.randn(input_size).astype(numpy.float32)
    filter = numpy.random.randn(17*17).astype(numpy.float32)
    args = [output, input, filter]

    tune_params = dict()
    tune_params["block_size_x"] = [16*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    kernel_tuner.tune_kernel("convolution_kernel", kernel_string,
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x, verbose=True)

You can try out this program from the ``examples`` directory in 
the kernel_tuner repository.


