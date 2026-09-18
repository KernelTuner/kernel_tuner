

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

The first paper on Kernel Tuner, please note that the capabilities of Kernel Tuner have significantly expanded since the first publication:

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

For referencing to Kernel Tuner's Bayesian Optimization strategy, please cite the following:

.. code:: latex

    @article{willemsen2021bayesian,
      author = {Willemsen, Floris-Jan and Van Nieuwpoort, Rob and Van Werkhoven, Ben},
      title = {Bayesian Optimization for auto-tuning GPU kernels},
      journal = {International Workshop on Performance Modeling, Benchmarking and Simulation
         of High Performance Computer Systems (PMBS) at Supercomputing (SC21)},
      year = {2021},
      url = {https://arxiv.org/abs/2111.14991}
    }


For a performance comparison of different optimization algorithms for auto-tuning and an analysis of tuning difficulty for different GPUs:
    
.. code:: latex

    @article{schoonhoven2022benchmarking,
      title={Benchmarking optimization algorithms for auto-tuning GPU kernels},
      author={Schoonhoven, Richard and van Werkhoven, Ben and Batenburg, K Joost},
      journal={IEEE Transactions on Evolutionary Computation},
      year={2022},
      publisher={IEEE}
    }


For referencing to Kernel Tuner's capabilities in measuring and optimizing energy consumption of GPU kernels, please cite the following:

.. code:: latex

    @article{schoonhoven2022going,
      author = {Schoonhoven, Richard and Veenboer, Bram, and van Werkhoven, Ben and Batenburg, K Joost},
      title = {Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning},
      journal = {International Workshop on Performance Modeling, Benchmarking and Simulation
         of High Performance Computer Systems (PMBS) at Supercomputing (SC22)},
      year = {2022},
      url = {https://arxiv.org/abs/2211.07260}
    }


For referencing to Kernel Tuner's capabilities super fast search space generation using constraint satisfaction problem solving, please cite the following:

.. code:: latex

    @inproceedings{willemsen2025efficient,
      title={Efficient construction of large search spaces for auto-tuning},
      author={Willemsen, Floris-Jan and van Nieuwpoort, Rob V and van Werkhoven, Ben},
      booktitle={Proceedings of the 54th International Conference on Parallel Processing},
      pages={668--677},
      year={2025}
    }


For referencing to Kernel Tuner's optimization algorithms hyperparameters and their tuning procedure, please cite the following:

.. code:: latex

    @inproceedings{willemsen2025tuning,
      title={Tuning the Tuner: Introducing Hyperparameter Optimization for Auto-Tuning},
      author={Willemsen, Floris-Jan and van Nieuwpoort, Rob V and van Werkhoven, Ben},
      booktitle={2025 IEEE International Conference on eScience (eScience)},
      pages={213--222},
      year={2025},
      organization={IEEE}
    }


For referencing to Kernel Tuner's optimization algorithm's capabilities to work with constrained search spaces, please cite the following:

.. code:: latex

    @inproceedings{willemsen2026constraint,
      title={Constraint-aware Optimization in Auto-Tuning},
      author={Willemsen, Floris-Jan and Heldens, Stijn and van Nieuwpoort, Rob V and van Werkhoven, Ben},
      booktitle={International Workshop on Automatic Performance Tuning (iWAPT)},
      year={2026}
    }


For referencing to Kernel Tuner's optimization algorithms that were automatically generated using LLMs, please cite the following:

.. code:: latex

    @inproceedings{willemsen2026automated,
      title={Automated Algorithm Design for Auto-Tuning Optimizers},
      author={Willemsen, Floris-Jan and van Stein, Niki and van Werkhoven, Ben},
      booktitle={Ninth Conference on Machine Learning and Systems},
      year={2026},
      url={https://openreview.net/forum?id=qKlHJCbY6m}
    }


For referencing to Kernel Tuner's capabilities in accuracy-aware tuning of mixed-precision GPU kernels, please cite the following:

.. code:: latex

    @article{heldens2026accuracy,
      title={Accuracy-Aware Mixed-Precision GPU Auto-Tuning},
      author={Heldens, Stijn and van Werkhoven, Ben},
      journal={IEEE Transactions on Parallel and Distributed Systems},
      year={2026},
      publisher={IEEE}
    }
