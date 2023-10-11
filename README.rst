Kernel Tuner
============

|Build Status| |CodeCov Badge| |PyPi Badge| |Zenodo Badge| |SonarCloud Badge| |OpenSSF Badge| |FairSoftware Badge|

Kernel Tuner simplifies the software development of optimized and auto-tuned GPU programs, by enabling Python-based unit testing of GPU code and making it easy to develop scripts for auto-tuning GPU kernels.
This also means no extensive changes and no new dependencies are required in the kernel code.
The kernels can still be compiled and used as normal from any host programming language.

Kernel Tuner provides a comprehensive solution for auto-tuning GPU programs, supporting auto-tuning of user-defined parameters in both host and device code, supporting output verification of all benchmarked kernels during tuning, as well as many optimization strategies to speed up the tuning process.

Documentation
-------------

The full documentation is available `here <https://kerneltuner.github.io/kernel_tuner/stable/index.html>`__.

Installation
------------

The easiest way to install the Kernel Tuner is using pip:

To tune CUDA kernels (`detailed instructions <https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda>`__):

- First, make sure you have the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ installed
- Then type: ``pip install kernel_tuner[cuda]``

To tune OpenCL kernels (`detailed instructions <https://kerneltuner.github.io/kernel_tuner/stable/install.html#opencl-and-pyopencl>`__):

- First, make sure you have an OpenCL compiler for your intended OpenCL platform
- Then type: ``pip install kernel_tuner[opencl]``

To tune HIP kernels (`detailed instructions <https://kerneltuner.github.io/kernel_tuner/stable/install.html#hip-and-pyhipl>`__):

- First, make sure you have an HIP runtime and compiler installed
- Then type: ``pip install kernel_tuner[hip]``

Or all:

- ``pip install kernel_tuner[cuda,opencl,hip]``

More information about how to install Kernel Tuner and its
dependencies can be found in the `installation guide
<http://kerneltuner.github.io/kernel_tuner/stable/install.html>`__.

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

The exact same Python code can be used to tune an OpenCL kernel:

.. code:: python

    kernel_string = """
    __kernel void vector_add(__global float *c, __global float *a, __global float *b, int n) {
        int i = get_global_id(0);
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

The Kernel Tuner will detect the kernel language and select the right compiler and
runtime. For every kernel in the parameter space, the Kernel Tuner will insert C
preprocessor defines for the tunable parameters, compile, and benchmark the kernel. The
timing results will be printed to the console, but are also returned by tune_kernel to
allow further analysis. Note that this is just the default behavior, what and how
tune_kernel does exactly is controlled through its many `optional arguments
<http://kerneltuner.github.io/kernel_tuner/stable/user-api.html#kernel_tuner.tune_kernel>`__.

You can find many - more extensive - example codes, in the
`examples directory <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/>`__
and in the `Kernel Tuner
documentation pages <http://kerneltuner.github.io/kernel_tuner/stable/index.html>`__.

Search strategies for tuning
----------------------------

Kernel Tuner supports many optimization algorithms to accelerate the auto-tuning process. Currently
implemented search algorithms are: Brute Force (default), Nelder-Mead, Powell, CG, BFGS, L-BFGS-B, TNC,
COBYLA, SLSQP, Random Search, Basinhopping, Differential Evolution, a Genetic Algorithm, Particle Swarm
Optimization, the Firefly Algorithm, Simulated Annealing, Dual Annealing, Iterative Local Search,
Multi-start Local Search, and Bayesian Optimization.

.. image:: https://github.com/KernelTuner/kernel_tuner/blob/master/doc/gemm-amd-summary.png?raw=true
    :width: 100%
    :align: center

Using a search strategy is easy, you only need to specify to ``tune_kernel`` which strategy and method
you would like to use, for example ``strategy="genetic_algorithm"`` or ``strategy="basinhopping"``.
For a full overview of the supported search strategies and methods please see the
Kernel Tuner documentation on `Optimization Strategies <https://kerneltuner.github.io/kernel_tuner/stable/optimization.html>`__.

Tuning host and kernel code
---------------------------

It is possible to tune for combinations of tunable parameters in
both host and kernel code. This allows for a number of powerfull things,
such as tuning the number of streams for a kernel that uses CUDA Streams
or OpenCL Command Queues to overlap transfers between host and device
with kernel execution. This can be done in combination with tuning the
parameters inside the kernel code. See the `convolution\_streams example
code <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/>`__
and the
`documentation <http://kerneltuner.github.io/kernel_tuner/stable/hostcode.html>`__
for a detailed explanation of the kernel tuner Python script.


Correctness verification
------------------------

Optionally, you can let the kernel tuner verify the output of every
kernel it compiles and benchmarks, by passing an ``answer`` list. This
list matches the list of arguments to the kernel, but contains the
expected output of the kernel. Input arguments are replaced with None.

.. code:: python

    answer = [a+b, None, None]  # the order matches the arguments (in args) to the kernel
    tune_kernel("vector_add", kernel_string, size, args, tune_params, answer=answer)

Contributing
------------

Please see the `Contributions Guide <http://kerneltuner.github.io/kernel_tuner/stable/contributing.html>`__.

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
      publisher={IEEE},
      url = {https://arxiv.org/abs/2210.01465}
    }

    @article{schoonhoven2022going,
      author = {Schoonhoven, Richard and Veenboer, Bram, and van Werkhoven, Ben and Batenburg, K Joost},
      title = {Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning},
      journal = {International Workshop on Performance Modeling, Benchmarking and Simulation
         of High Performance Computer Systems (PMBS) at Supercomputing (SC22)},
      year = {2022},
      url = {https://arxiv.org/abs/2211.07260}
    }


.. |Build Status| image:: https://github.com/KernelTuner/kernel_tuner/actions/workflows/build-test-python-package.yml/badge.svg
   :target: https://github.com/KernelTuner/kernel_tuner/actions/workflows/build-test-python-package.yml
.. |CodeCov Badge| image:: https://codecov.io/gh/KernelTuner/kernel_tuner/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/KernelTuner/kernel_tuner
.. |PyPi Badge| image:: https://img.shields.io/pypi/v/kernel_tuner.svg?colorB=blue
   :target: https://pypi.python.org/pypi/kernel_tuner/
.. |Zenodo Badge| image:: https://zenodo.org/badge/54894320.svg
   :target: https://zenodo.org/badge/latestdoi/54894320
.. |SonarCloud Badge| image:: https://sonarcloud.io/api/project_badges/measure?project=KernelTuner_kernel_tuner&metric=alert_status
   :target: https://sonarcloud.io/dashboard?id=KernelTuner_kernel_tuner
.. |OpenSSF Badge| image:: https://bestpractices.coreinfrastructure.org/projects/6573/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/6573
.. |FairSoftware Badge| image:: https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green
   :target: https://fair-software.eu
