

The Kernel Tuner documentation
==============================

Kernel Tuner is a software development tool for the creation of highly-optimized and tuned GPU applications.

The Kernel Tuner documentation pages are mostly about Kernel Tuner itself, but there are a number of related repositories that 
are considered part of the Kernel Tuner family:

 * `Kernel Tuner Tutorial <https://github.com/KernelTuner/kernel_tuner_tutorial>`__
 * `Kernel Launcher <https://github.com/KernelTuner/kernel_launcher>`__
 * `KT Dashboard <https://github.com/KernelTuner/dashboard>`__

Quick install
-------------

The easiest way to install the Kernel Tuner is using pip:

To tune CUDA kernels:

- First, make sure you have the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ installed
- Then type: ``pip install kernel_tuner[cuda]``

To tune OpenCL kernels:

- First, make sure you have an OpenCL compiler for your intended OpenCL platform
- Then type: ``pip install kernel_tuner[opencl]``

To tune HIP kernels:

- First, make sure you have an HIP runtime and compiler installed
- Then type: ``pip install kernel_tuner[hip]``

Or all:

- ``pip install kernel_tuner[cuda,opencl,hip]``

More information about how to install Kernel Tuner and its
dependencies can be found under :ref:`install`. 

Example usage
-------------

The following shows a simple example for tuning a CUDA kernel:

.. code:: python

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 10000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)
    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128, 256, 512]

    tune_kernel("vector_add", kernel_string, size, args, tune_params)


Citation
--------
If you use Kernel Tuner in research or research software, please cite the most relevant among the following publications:

.. code:: latex

    @article{kerneltuner,
      author  = {Ben van Werkhoven},
      title   = {Kernel Tuner: A search-optimizing GPU code auto-tuner},
      journal = {Future Generation Computer Systems},
      year = {2019},
      volume  = {90},
      pages = {347-358},
      url = {https://www.sciencedirect.com/science/article/pii/S0167739X18313359},
      doi = {https://doi.org/10.1016/j.future.2018.08.004}
    }

    @article{willemsen2021bayesian,
      author = {Willemsen, Floris-Jan and Van Nieuwpoort, Rob and Van Werkhoven, Ben},
      title = {Bayesian Optimization for auto-tuning GPU kernels},
      journal = {International Workshop on Performance Modeling, Benchmarking and Simulation
         of High Performance Computer Systems (PMBS) at Supercomputing (SC21)},
      year = {2021},
      url = {https://arxiv.org/abs/2111.14991}
    }
    
    @article{schoonhoven2022benchmarking,
      title={Benchmarking optimization algorithms for auto-tuning GPU kernels},
      author={Schoonhoven, Richard and van Werkhoven, Ben and Batenburg, K Joost},
      journal={IEEE Transactions on Evolutionary Computation},
      year={2022},
      publisher={IEEE}
    }

    @article{schoonhoven2022going,
      author = {Schoonhoven, Richard and Veenboer, Bram, and van Werkhoven, Ben and Batenburg, K Joost},
      title = {Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning},
      journal = {International Workshop on Performance Modeling, Benchmarking and Simulation
         of High Performance Computer Systems (PMBS) at Supercomputing (SC22)},
      year = {2022},
      url = {https://arxiv.org/abs/2211.07260}
    }
