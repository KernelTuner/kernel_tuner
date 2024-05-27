Using structs
-------------

One of the issues with calling GPU kernels from Python is the use of custom data types in kernel arguments. In general, it is recommended for portability of your GPU code, which may be
used in any host program in any host programming language, to keep the interface of your kernels as simple as possible. This means sticking to simple pointers of primitive types such as integer, float, and double. 
For performance reasons, it is also recommended to not use arrays of structs for kernel arguments, as this is very likely to lead to inefficient memory accesses on the GPU.

However, there are situations, in particular in scientific applications, where the GPU code needs a lot of input parameters where it makes sense to collect these in a struct that 
describes the simulation or experimental setup. For these use cases, it is possible to use Python's built-in ``struct`` library, in particular the function ``struct.pack()``. For how to use 
``struct.pack``, please consult the `Python documentation <https://docs.python.org/3/library/struct.html>`__. In the code below we show part of Python script that uses ``struct.pack``, 
Numpy, and Kernel Tuner to call a CUDA kernel that uses a struct as kernel argument.


.. code:: python

    import struct
    import numpy as np
    import kernel_tuner as kt

    def create_receive_spec_struct():
        ...

        # Use struct.pack to create a byte representation of our struct
        # The format string uses:
        #   i for integer, P for Pointer, f for float, ? for bool
        # The 0l at the end ensures padding to the next long (8bytes)
        packstr = struct.pack('iiiiiiiiiiiPPi?fffi?0l', 
                              nSamples, nSamplesIQ, nSlowTimeSamples,
                              nChannels, nTX, nRepeats, nFastTimeSamples,
                              rfSize, mNRows, mNRowsIQ, nActiveChannels,
                              0, 0, 0, isIQ, Fs, FsIQ, Fc, nBuffers,
                              initialized)
        return np.frombuffer(np.array(packstr),
                             np.dtype((np.void, len(packstr))))[0]

    receive_spec = create_receive_spec_struct()

    ...

    args = [bf, rf, receive_spec, recon]

    kt.tune_kernel(kernel_name, kernel_source, problem_size, args, tune_params)


The most difficult part of this code is ensuring the struct.pack format string is correct and keeping it in sync with the GPU code. Note the ``0l`` at the end of string. This enables 
padding to the next long, which is sometimes needed when the struct argument is not the last argument in the kernel argument list.

Using the packstr returned by struct.pack, we can create a numpy buffer, in this case we are only interested in the first element in the array, hence the ``[0]`` at the end. To tell Numpy 
what data type our packstr hold we specify the dtype as an np.void of length ``len(packstr)``, which is the number of bytes in our data type. If you need an array of structs, you only 
need some slight modifications to the example code above.
