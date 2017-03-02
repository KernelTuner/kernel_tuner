.. highlight:: python
    :linenothreshold: 5

Tutorial
========

This tutorial will introduce you to everything you need to know to start 
tuning your own kernels.

You've probably seen the rather minimalistic vector add example, which is
a bit too simplistic to fully tell you how to tune any given kernel. 
Therefore, this tutorial starts out with a little bit more complex, yet 
still quite simple, 2D stencil kernel written in CUDA.

.. note:: 

    If you prefer OpenCL over CUDA, don't worry. Everything in this tutorial 
    applies as much to OpenCL as it does to CUDA. But I will use CUDA code in 
    the examples, and CUDA terminology in the text. 

    Here's a quick translation guide:
    An OpenCL work item is a called a thread in CUDA, a work group is called a 
    thread block, and an NDRange is called a grid. Instead of OpenCL's 
    get_local_id() and get_group_id(), CUDA uses built-in variables threadIdx 
    and blockIdx. 

Tuning a 2D stencil kernel
--------------------------

We use a 2D stencil kernel as an example kernel to get you started with writing 
your Python scripts and start tuning with the Kernel Tuner. 2D stencil kernels 
like the one we use here are an compute-intensive part of iterative solvers 
that are used by many applications that simulate physical processes, like 
diffusion. Let's say you have written a CUDA kernel to perform the 2D stencil 
computation on the GPU, like the one shown below.

Like in any CUDA kernel, you as a programmer have to decide how to group your 
threads into thread blocks. And like in many CUDA kernels, the thread block 
size that we choose for our 2D stencil kernel is not really that important for 
the output of the kernel. However, the thread block dimensions will have an 
impact on the performance of your kernels. And the optimal setting will be 
different for different GPUs.

So how do you know which thread block size to choose? Simply try them all with
auto tuning!

.. code-block:: c

    #define domain_width    500
    #define domain_height   500

    __global__ void stencil_kernel(float *x_new, float *x_old) {
        int x = blockIdx.x * block_size_x + threadIdx.x;
        int y = blockIdx.y * block_size_y + threadIdx.y;

        if (y>0 && y<domain_height-1 && x>0 && x<domain_width-1) {

        x_new[y*domain_width+x] = ( x_old[ (y  ) * domain_width + (x  ) ] +
                                    x_old[ (y  ) * domain_width + (x-1) ] +
                                    x_old[ (y  ) * domain_width + (x+1) ] +
                                    x_old[ (y+1) * domain_width + (x  ) ] +
                                    x_old[ (y-1) * domain_width + (x  ) ] ) / 5.0f;

        }
    }

This 2D stencil kernel assumes that a thread will be created for each element 
in the domain. Each thread then simply takes the average of the element 
corresponding with its computed thread index in ``x_old`` and its four direct 
neighbors, one in every direction. The newly computed value is then stored in 
``x_new``. Iterative solvers will have to call kernels like this one many 
times, so it is important that this kernel is efficient.

You may notice that the kernel uses two, currently undefined, constants 
``block_size_x`` and ``block_size_y`` instead of built-in variables blockDim.x 
or blockDim.y. Setting the thread block dimensions at compile-time is often a 
good idea for performance. If you don't need to vary the thread block size at 
run-time, the compiler can, for example, unroll loops that iterate using the 
thread block size.

Let's take a look at how we can write a small Python script that uses the 
Kernel Tuner to test the performance of our kernel for different combinations
of ``block_size_x`` and ``block_size_y``.

Setup tuning parameters
~~~~~~~~~~~~~~~~~~~~~~~

We call parameters in the kernel, like ``block_size_x`` and ``block_size_y``,
tunable parameters. This is because we want to tune the performance of the 
kernel based on the values given to these parameters.

To tell the Kernel Tuner about our tunable parameters we use a Python 
dictionary, which is basically a hashmap. For every tunable parameter, we 
create a key-value pair in the dictionary. The key is the name of the parameter 
as a string. The value associated with that key is a list of possible values 
for the tunable parameter.

Let's look at an example:

.. code-block:: python

    tune_params = dict()
    tune_params["block_size_x"] = [32*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

Now just in case you are not a Python guru, an expression between square 
brackets ``[ ]`` is a list comprehension. ``[32*i for i in 
range(1,9)]`` will create a list of multiples of 32 ranging from 
``32*1`` up to ``32*8``. For ``block_size_y`` we creates a list of 
powers of 2 ranging from ``2**0 = 1`` up to ``2**5 = 32``.

The values that we have picked here are just examples, you can basically 
pick any list of values that you like. The Kernel Tuner will check the 
maximum number of threads per thread block supported by your GPU at 
run-time, and automatically skip over kernel configurations that attempt 
to use more. The Kernel Tuner will do this silently, unless you use the 
option ``verbose=True``.

While the Kernel Tuner allows you to pick any value that you like, an 
experienced CUDA programmer will know that only certain values will make sense. 
For example, a thread block size that is a multiple of 32 is likely to give 
better performance, because threads in CUDA are scheduled in warps of 32 
threads.

For each kernel that the Kernel Tuner benchmarks, it will prepend the 
source code with C preprocessor directives to define all tuning 
parameters and their current value, for example:

.. code-block:: c

    #define block_size_x 32
    #define block_size_y 1

There is of course much more that you can tune within a kernel than just 
the thread block dimensions. Basically, you are completely free to write 
code that uses C preprocessor directives to change its behavior. If you
tell the Kernel Tuner about all the possible values for this parameter, it
will then benchmark all of possible execution paths in your 
code. However, the Kernel Tuner currently uses the convention that 
``block_size_x``, ``block_size_y``, and ``block_size_z`` are used for 
specifying the thread block dimensions.

If you want to be able to compile your code when not using the Kernel 
Tuner, you can simply add preprocessor directives for providing default 
values to all tunable parameters, for example to the beginning our
2D stencil kernel we could add:

.. code-block:: c

    #ifndef block_size_x
        #define block_size_x 16
    #endif
    #ifndef block_size_y
        #define block_size_y 16
    #endif

To ensure that the kernel code can be compiled directly by any CUDA compiler, 
even when the Kernel Tuner is not used.


Calling tune_kernel()
~~~~~~~~~~~~~~~~~~~~~

Now that we've setup our tuning parameters it is time to look at how to call 
the Kernel Tuner, and most importantly ``tune_kernel()``.

.. function:: tune_kernel(kernel_name, kernel_string, problem_size, arguments, tune_params, grid_div_x=None, grid_div_y=None, restrictions=None, answer=None, atol=1e-6, verbose=False, lang=None, device=0, platform=0, cmem_args=None, sample=False, compiler_options=None, log=None)

As you can see, there are a lot optional parameters and we're not going 
to cover all of them right now, if you're interested check out the
:ref:`details`. Let's start with the basics. The Kernel Tuner has to 
know at least the following things:

* ``kernel_name``: The name of the kernel
* ``kernel_string``: The source code that contains that kernel
* ``problem_size``: The domain over which you create threads and thread blocks
* ``arguments``: The arguments to use when calling the kernel
* ``tune_params``: The dictionary with tunable parameters

So let's assume that our 2D stencil kernel is stored in the file 
``stencil.cu``. We can read its contents into a string, so that we can 
later pass it to the Kernel Tuner.

.. code-block:: python

    with open('stencil.cu', 'r') as f:
        kernel_string = f.read()

Now we can use Numpy to generate some random input data, and create a 
list of arguments that matches the argument list of our 
``stencil_kernel`` function written in CUDA. It is important that the 
order and type matches the function specification of our kernel.

.. code-block:: python

    problem_size = (500, 500)
    size = numpy.prod(problem_size)

    x_old = numpy.random.randn(size).astype(numpy.float32)
    x_new = numpy.copy(x_old)
    args = [x_new, x_old]

Note that `size` matches the size of the domain in used in the CUDA code.
Moreover, we use ``astype`` to ensure that the Numpy array consists of 32-bit floating-point values, 
as expected by our CUDA kernel.

Instead of generating random data you can of course also use data from a 
file, Python offers many convenient functions for this, for example take
look at numpy.fromfile() or numpy.loadtxt().

The list named `args` will be used as the argument list that we'll pass
to tune_kernel. The Kernel Tuner requires that list contains only Numpy arrays
or Numpy scalar values, and that the order of arguments matches that of the CUDA kernel.

Now we are almost ready to call tune_kernel(). However, we have not told 
the Kernel Tuner anything about how many thread blocks should be created 
to launch the kernel. If you do not specify this the Kernel Tuner will 
assume a default way for computing the number of thread blocks. The grid 
dimension in the x-direction will default to ``problem_size[0] / 
block_size_x`` and the y-direction will default to ``1``. This is 
convenient for small 1D kernels like vector add, but for our 2D stencil 
kernel we need to specify how the tuning parameters divide our problem 
size.

You can tell the Kernel Tuner how to determine the number of blocks it
should create through the so called grid divisor lists, which you can 
specify using the optional arguments ``grid_div_x`` and ``grid_div_y``.
Now let's look at an example of how to setup these grid divisor lists.

So for our 2D stencil kernel, we have a 2D domain over which we want to 
create threads and thread blocks in a way that we create one thread for 
each element in the domain. So to get to the number of thread blocks, 
the Kernel Tuner should just divide the problem_size in a particular 
dimension with the thread block size in that dimension. Therefore, we 
specify the following:

.. code-block:: python

    grid_div_x = ["block_size_x"]
    grid_div_y = ["block_size_y"]

Note that these are lists, you can add multiple tuning parameters to the 
list. If you want to, you can even write arithmetic expressions within 
these strings. The Kernel Tuner will evaluate all strings and multiply 
them together. Then it will use this product to divide the problem size 
in that dimension **rounded up**.

Putting it all together
~~~~~~~~~~~~~~~~~~~~~~~

Now let's put everything that we've gone through together in a Python script
so you can try it out and see what it does.

.. code-block:: python

    import numpy
    import kernel_tuner

    with open('stencil.cu', 'r') as f:
        kernel_string = f.read()

    problem_size = (4096, 2048)
    size = numpy.prod(problem_size)

    x_old = numpy.random.randn(size).astype(numpy.float32)
    x_new = numpy.copy(x_old)
    args = [x_new, x_old]

    tune_params = dict()
    tune_params["block_size_x"] = [32*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    grid_div_x = ["block_size_x"]
    grid_div_y = ["block_size_y"]

    kernel_tuner.tune_kernel("stencil_kernel", kernel_string, problem_size,
        args, tune_params, grid_div_x=grid_div_x, grid_div_y=grid_div_y,
        verbose = True)






