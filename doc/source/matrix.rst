.. highlight:: python
    :linenothreshold: 5


Matrix Multiplication
---------------------

Matrix multiplication is one of the most well-known linear algebra 
algorithms, and frequently used to demonstrate the high-performance 
computing capabilities of GPUs. As such, an example using matrix 
multiplication could not be left out. A naive CUDA kernel for 
a square matrix multiplication is:

.. code-block:: c

    __global__ void matmul_kernel(float *C, float *A, float *B) {
        int x = blockIdx.x * block_size_x + threadIdx.x;
        int y = blockIdx.y * block_size_y + threadIdx.y;
        float sum = 0.0;

        for (int k=0; k<WIDTH; k++) {
            sum += A[y*WIDTH+k] * B[(y+i)*WIDTH+x];
        }

        C[y*WIDTH+x] = sum;
    }

This kernel simply creates a single thread per output element. Each 
thread computes the index of the element it is responsible for, and 
iterates over the corresponding row in A, and corresponding column in B.

There aren't many parameters to tune yet, and more importantly, tuning 
will not be very effective because this kernel will be limited by 
bandwidth rather than compute. There is however, a lot of opportunity 
for data reuse, which is realized by making the threads in a thread 
block collaborate.

Increase data reuse
~~~~~~~~~~~~~~~~~~~

This can be solved by using a technique called loop-blocking or 
loop-tiling. We define two square data structures in `shared memory`, 
which will be used for storing square parts of matrix A and B. The 
threads in a thread block will collaboratively fill these two variables, 
and then proceed to perform all the computations that need this data, 
before moving to the next blocked iteration.

.. code-block:: c

    __global__ void matmul_kernel(float *C, float *A, float *B) {

        __shared__ float sA[block_size][block_size];
        __shared__ float sB[block_size][block_size];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x = blockIdx.x * block_size + tx;
        int y = blockIdx.y * block_size + ty;

        float sum = 0.0;
        int k,kb;

        for (k=0; k<WIDTH; k+=block_size) {
            __synchthreads();
            sA[ty][tx] = A[y*WIDTH+k+tx];
            sB[ty][tx] = B[(k+ty)*WIDTH+x];
            __synchthreads();

            for (kb=0; kb<block_size; kb++) {
                sum += sA[ty][kb] * sB[kb][tx];
            }

        }

        C[y*WIDTH+x] = sum;
    }

This kernel drastically reduces memory bandwidth consumption and should 
be compute bound rather than memory-bandwidth bound for most combinations of 
matrix sizes and thread block sizes.

However, there still is not too much that can be tuned in this kernel. 
In fact, because the thread block size needs to be a square, there only 
a handful of configurations we can try. Fortunately, we can add serveral 
more optimizations to the code that also open the parameter space for 
tuning.

Increase work per thread
~~~~~~~~~~~~~~~~~~~~~~~~

We will use two different forms of 1xN tiling in this example:

First of all, in the x-direction we will use tiling in a way that is 
similar to the convolution example. The area of output data that is
processed by a single thread block is increased by a factor of N,
and as such shared memory usage also increases by a factor N.
This means that the number of thread blocks needed to execute 
the kernel for this problem size is reduced by a factor of N,
where N is the tiling factor. 
While this may reduce occupancy due to increased shared memory and
register usage, this optimization drastically reduces 
the number of redundant instructions that were previously distributed 
across multiple thread blocks.

Secondly, in the y-direction we will use a different form of 1xN tiling, 
where we tile within the thread block. This too means that threads will 
compute multiple elements, but in this case, not the total number of thread 
blocks is reduced, but instead the number of threads per block goes down.

Note that these two different forms of tiling could have combined in 
different or even multiple ways to increase the tuning parameter space 
even further. However, for the purposes of this example, the resulting 
kernel is already complex enough:

.. code-block:: c

    __global__ void matmul_kernel(float *C, float *A, float *B) {

        __shared__ float sA[block_size_y*tile_size_y][block_size_x];
        __shared__ float sB[block_size_y*tile_size_y][block_size_x * tile_size_x];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x = blockIdx.x * block_size_x * tile_size_x + threadIdx.x;
        int y = blockIdx.y * block_size_y * tile_size_y + threadIdx.y;
        int k, kb;

        float sum[tile_size_y][tile_size_x];

        for (k = 0; k < WIDTH; k += block_size_x) {

            __syncthreads ();
            #pragma unroll
            for (int i = 0; i < tile_size_y; i++) {
                sA[ty + block_size_y * i][tx] = A[y * WIDTH + block_size_y * i * WIDTH + k + tx];

                #pragma unroll
                for (int j = 0; j < tile_size_x; j++) {
                    sB[ty + block_size_y * i][tx + j * block_size_x] = B[(k + ty + block_size_y * i) * WIDTH + x + j * block_size_x];
                }
            }
            __syncthreads ();

            //compute
            #pragma unroll
            for (kb = 0; kb < block_size_x; kb++) {

                #pragma unroll
                for (int i = 0; i < tile_size_y; i++) {
                    #pragma unroll
                    for (int j = 0; j < tile_size_x; j++) {
	                    sum[i][j] += sA[ty + block_size_y * i][kb] * sB[kb][tx + j * block_size_x];
	                }
                }

            }

        }

        //store result
        #pragma unroll
        for (int i = 0; i < tile_size_y; i++) {
            #pragma unroll
            for (int j = 0; j < tile_size_x; j++) {
                C[y * WIDTH + x + block_size_y * i * WIDTH + j * block_size_x] = sum[i][j];
            }
        }

    }


Setup tuning parameters
~~~~~~~~~~~~~~~~~~~~~~~

Now we will explain how to use the kernel_tuner to tune all the 
parameters of this highly-flexible implementation. We'll first show the 
Python script and then explain it step-by-step.

::

    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)

    A = numpy.random.randn(size).astype(numpy.float32)
    B = numpy.random.randn(size).astype(numpy.float32)
    C = numpy.zeros_like(A)
    args = [C, A, B]

    tune_params = dict()
    tune_params["block_size_x"] = [16*2**i for i in range(3)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_params["tile_size_x"] = [2**i for i in range(4)]
    tune_params["tile_size_y"] = [2**i for i in range(4)]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    restrict = ["block_size_x==block_size_y*tile_size_y"]

    kernel_tuner.tune_kernel("matmul_kernel", kernel_string,
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x,
        restrictions=restrict, verbose=True)

As usual we first setup the kernel arguments and problem size so that 
the kernel_tuner knows how to call the kernel. Then for the 
``block_size_x`` we choose a range of thread block sizes that seem 
reasonable, in this case ``[16, 32, 64]``. You typically want the total 
number of threads within a thread block to be a multiple of 32 
(warpsize) or even 64 (number of register banks on some cards). And 
because the tiling factors will increase the amount of work per thread 
block, as well as the amount of shared memory used we start a tad 
conservatively here. For ``block_size_y``, and the tiling factors in both 
directions, we just pick a range of powers of two.

Now let's fast-forward to the interesting bit: Remember that the area 
operated on by the thread block should be a square. In this kernel 
however, we allow ``block_size_x`` and ``block_size_y`` to vary 
independently, while ``tile_size_y`` increases the amount of work per 
thread in the y-direction within the thread block. This yields a 
discontinuous search space in which only part of the configurations are 
actually valid. Therefore we use the ``restrictions`` optional argument of 
``tune_kernel``.

``restrictions`` expects a list of strings that contain a boolean 
expression that may use the tuning parameters as variables. Any 
occurences of tuning parameter names will be replaced with the specific 
value of this parameter when the kernel configuration is evaluated. All 
expressions in the list passed as restrictions need to evaluate to 
``True`` for the configuration to be considered valid and therefore part
of the parameter space.








