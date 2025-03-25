

<div align="center">
  <img width="500px" src="https://raw.githubusercontent.com/KernelTuner/kernel_tuner/master/doc/images/KernelTuner-logo.png"/>
</div>

---
[![Build Status](https://github.com/KernelTuner/kernel_tuner/actions/workflows/test-python-package.yml/badge.svg)](https://github.com/KernelTuner/kernel_tuner/actions/workflows/test-python-package.yml)
[![CodeCov Badge](https://codecov.io/gh/KernelTuner/kernel_tuner/branch/master/graph/badge.svg)](https://codecov.io/gh/KernelTuner/kernel_tuner)
[![PyPi Badge](https://img.shields.io/pypi/v/kernel_tuner.svg?colorB=blue)](https://pypi.python.org/pypi/kernel_tuner/)
[![Zenodo Badge](https://zenodo.org/badge/54894320.svg)](https://zenodo.org/badge/latestdoi/54894320)
[![SonarCloud Badge](https://sonarcloud.io/api/project_badges/measure?project=KernelTuner_kernel_tuner&metric=alert_status)](https://sonarcloud.io/dashboard?id=KernelTuner_kernel_tuner)
[![OpenSSF Badge](https://bestpractices.coreinfrastructure.org/projects/6573/badge)](https://bestpractices.coreinfrastructure.org/projects/6573)
[![FairSoftware Badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
---


Create optimized GPU applications in any mainstream GPU 
programming language (CUDA, HIP, OpenCL, OpenACC).

What Kernel Tuner does:

- Works as an external tool to benchmark and optimize GPU kernels in isolation
- Can be used directly on existing kernel code without extensive changes 
- Can be used with applications in any host programming language
- Blazing fast search space construction
- More than 20 [optimization algorithms](https://kerneltuner.github.io/kernel_tuner/stable/optimization.html) to speedup tuning
- Energy measurements and optimizations [(power capping, clock frequency tuning)](https://arxiv.org/abs/2211.07260)
- ... and much more! For example, [caching](https://kerneltuner.github.io/kernel_tuner/stable/cache_files.html), [output verification](https://kerneltuner.github.io/kernel_tuner/stable/correctness.html), [tuning host and device code](https://kerneltuner.github.io/kernel_tuner/stable/hostcode.html), [user defined metrics](https://kerneltuner.github.io/kernel_tuner/stable/metrics.html), see [the full documentation](https://kerneltuner.github.io/kernel_tuner/stable/index.html).



## Installation

- First, make sure you have your [CUDA](https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda), [OpenCL](https://kerneltuner.github.io/kernel_tuner/stable/install.html#opencl-and-pyopencl), or [HIP](https://kerneltuner.github.io/kernel_tuner/stable/install.html#hip-and-hip-python) compiler installed
- Then type: `pip install kernel_tuner[cuda]`, `pip install kernel_tuner[opencl]`, or `pip install kernel_tuner[hip]`
- or why not all of them: `pip install kernel_tuner[cuda,opencl,hip]`

More information on installation, also for other languages, in the [installation guide](http://kerneltuner.github.io/kernel_tuner/stable/install.html).

## Example

```python
import numpy as np
from kernel_tuner import tune_kernel

kernel_string = """
__global__ void vector_add(float *c, float *a, float *b, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
"""

n = np.int32(10000000)

a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros_like(a)

args = [c, a, b, n]

tune_params = {"block_size_x": [32, 64, 128, 256, 512]}

tune_kernel("vector_add", kernel_string, n, args, tune_params)
```

More [examples here](https://kerneltuner.github.io/kernel_tuner/stable/examples.html).

## Resources

- [Full documentation](https://kerneltuner.github.io/kernel_tuner/stable/)
- Guides:
  - [Getting Started](https://kerneltuner.github.io/kernel_tuner/stable/quickstart.html)
  - [Convolution](https://kerneltuner.github.io/kernel_tuner/stable/convolution.html)
  - [Diffusion](https://kerneltuner.github.io/kernel_tuner/stable/diffusion.html)
  - [Matrix Multiplication](https://kerneltuner.github.io/kernel_tuner/stable/matrix_multiplication.html)
- Features & Use cases:
  - [Full list of examples](https://kerneltuner.github.io/kernel_tuner/stable/examples.html)
  - [Output verification](https://kerneltuner.github.io/kernel_tuner/stable/correctness.html)
  - [Test GPU code from Python](https://github.com/KernelTuner/kernel_tuner/blob/master/examples/cuda/test_vector_add.py)
  - [Tune code in both host and device code](https://kerneltuner.github.io/kernel_tuner/stable/hostcode.html)
  - [Optimization algorithms](https://kerneltuner.github.io/kernel_tuner/stable/optimization.html)
  - [Mixed-precision & Accuracy tuning](https://github.com/KernelTuner/kernel_tuner/blob/master/examples/cuda/accuracy.py)
  - [Custom metrics & tuning objectives](https://kerneltuner.github.io/kernel_tuner/stable/metrics.html)
- **Kernel Tuner Tutorial** slides [[PDF]](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/slides/2022_SURF/SURF22-Kernel-Tuner-Tutorial.pdf), hands-on:
  - Vector add example [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/00_Kernel_Tuner_Introduction.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/00_Kernel_Tuner_Introduction.ipynb)
  - Tuning thread block dimensions [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/01_Kernel_Tuner_Getting_Started.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/01_Kernel_Tuner_Getting_Started.ipynb)
  - Search space restrictions & output verification [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/02_Kernel_Tuner_Intermediate.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/02_Kernel_Tuner_Intermediate.ipynb)
  - Visualization & search space optimization [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/03_Kernel_Tuner_Advanced.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/03_Kernel_Tuner_Advanced.ipynb)
- **Energy Efficient GPU Computing** tutorial slides [[PDF]](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/slides/2023_Supercomputing/SC23.pdf), hands-on:
  - Kernel Tuner for GPU energy measurements [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/energy/00_Kernel_Tuner_Introduction.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/energy/00_Kernel_Tuner_Introduction.ipynb)
  - Code optimizations for energy [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/energy/01_Code_Optimizations_for_Energy.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/energy/01_Code_Optimizations_for_Energy.ipynb)
  - Mixed precision and accuracy tuning [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/energy/02_Mixed_precision_programming.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/energy/02_Mixed_precision_programming.ipynb)
  - Optimzing for time vs for energy [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/energy/03_energy_efficient_computing.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/energy/03_energy_efficient_computing.ipynb)


## Kernel Tuner ecosystem

<img width="250px" src="https://raw.githubusercontent.com/KernelTuner/kernel_tuner/master/doc/images/kernel_launcher.png"/><br />C++ magic to integrate auto-tuned kernels into C++ applications 

<img width="250px" src="https://raw.githubusercontent.com/KernelTuner/kernel_tuner/master/doc/images/kernel_float.png"/><br />C++ data types for mixed-precision CUDA kernel programming

<img width="275px" src="https://raw.githubusercontent.com/KernelTuner/kernel_tuner/master/doc/images/kernel_dashboard.png"/><br />Monitor, analyze, and visualize auto-tuning runs


## Communication & Contribution

- GitHub [Issues](https://github.com/KernelTuner/kernel_tuner/issues): Bug reports, install issues, feature requests, work in progress
- GitHub [Discussion group](https://github.com/orgs/KernelTuner/discussions): General questions, Q&A, thoughts

Contributions are welcome! For feature requests, bug reports, or usage problems, please feel free to create an issue.
For more extensive contributions, check the [contribution guide](http://kerneltuner.github.io/kernel_tuner/stable/contributing.html).

## Citation

If you use Kernel Tuner in research or research software, please cite the most relevant among the [publications on Kernel 
Tuner](https://kerneltuner.github.io/kernel_tuner/stable/#citation). To refer to the project as a whole, please cite:

```latex
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
```

