.. _installation:

Installation
============

The Kernel Tuner requires several packages to be installed. First of all, you need a
working Python version, several Python packages, and optionally CUDA and/or OpenCL
installations. All of this is explained in detail in this guide.

For comprehensive step-by-step instructions on setting up a development environment, see :ref:`Development Environment <dev-environment>`.

Python
------

You need a Python installation. We recommend using Python 3 and installing it with `Miniconda <https://conda.io/miniconda.html>`__.
Linux users could type the following to download and install Python 3 using Miniconda:

.. code-block:: bash

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

You are of course also free to use your own Python installation, and the Kernel Tuner is developed to be fully compatible with Python 3.10 and newer.

Installing Python Packages
--------------------------

Note that when you are using a native Python installation, the `pip` command used 
to install
Kernel Tuner and its dependencies requires `sudo` rights for system wide installation. 

Sudo rights are typically not required when using Miniconda or virtual environments.
You could also use e.g. the `--user` or `--prefix` option of `pip` to install into
your home directory,
this requires that your home directory is on your `$PYTHONPATH` environment variable
(see for further details the pip documentation).

The following command will install Kernel Tuner together with the required dependencies:

.. code-block:: bash

    pip install kernel_tuner

There are also optional dependencies, explained below.


.. _installing cuda:

CUDA and PyCUDA
---------------

Installing CUDA and PyCUDA is optional, because you may want to only use Kernel
Tuner for tuning OpenCL or C kernels.

If you want to use the Kernel Tuner to tune
CUDA kernels you will first need to install the CUDA toolkit
(https://developer.nvidia.com/cuda-toolkit). A recent version of the
CUDA toolkit (and the PyCUDA Python bindings for CUDA) are
recommended (older version may work, but may not support all features of
Kernel Tuner).

It's very important that you install the CUDA toolkit before trying to install PyCuda.

You can install PyCuda manually using:

.. code-block:: bash

    pip install pycuda

Or you could install Kernel Tuner and PyCUDA together if you haven't done so already:

.. code-block:: bash

    pip install kernel_tuner[cuda]

If you run into trouble with installing PyCuda, make sure you have CUDA installed first.
Also make sure that the Python package Numpy is already installed, e.g. using `pip install numpy`.

If you retry the ``pip install pycuda`` command, you may need to use the
``--no-cache-dir`` option to ensure the pycuda installation really starts over and not continues
from an installation that is failing.

If this fails, I recommend to see the PyCuda installation guide (https://wiki.tiker.net/PyCuda/Installation)


Other CUDA Backends
-------------------

Kernel Tuner can also be used with CuPy (https://cupy.dev/) or Nvidia's CUDA Python bindings (https://nvidia.github.io/cuda-python/). Please see the installation instructions of those projects for how the required Python packages.

Please refer to the documentation on `backends <https://kerneltuner.github.io/kernel_tuner/stable/backends.html>`__ on how to use and select these backends.



OpenCL and PyOpenCL
-------------------

Before we can install PyOpenCL you'll need an OpenCL compiler. There are several
OpenCL compilers available depending on the OpenCL platform you want to your
code to run on.

* `AMD APP SDK <https://rocmdocs.amd.com/en/latest/Programming_Guides/Opencl-programming-guide.html>`__
* `Intel OpenCL <https://software.intel.com/en-us/iocl_rt_ref>`__
* `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`__
* `Apple OpenCL <https://developer.apple.com/opencl/>`__
* `Beignet <https://www.freedesktop.org/wiki/Software/Beignet/>`__

You can also look at this `OpenCL Installation Guide <https://wiki.tiker.net/OpenCLHowTo>`__ for PyOpenCL.

As with the CUDA toolkit, recent versions of one or more of the above OpenCL SDK's and
PyOpenCL are recommended to support all features of the Kernel Tuner.

After you've installed your OpenCL compiler of choice you can install PyOpenCL using:

.. code-block:: bash

    pip install pyopencl

Or you could install Kernel Tuner and PyOpenCL together if you haven't done so already:

.. code-block:: bash

    pip install kernel_tuner[opencl]

If this fails, please see the PyOpenCL installation guide (https://wiki.tiker.net/PyOpenCL/Installation)

HIP and HIP Python
------------------

Before we can install HIP Python, you'll need to have the HIP runtime and compiler installed on your system.
The HIP compiler is included as part of the ROCm software stack. Here is AMD's installation guide:

* `ROCm Documentation: HIP Installation Guide <https://docs.amd.com/bundle/HIP-Installation-Guide-v5.3/page/Introduction_to_HIP_Installation_Guide.html>`__

After you've installed HIP, you will need to install HIP Python. Run the following command in your terminal to install:

First identify the first three digits of the version number of your ROCm™ installation.
Then install the HIP Python package(s) as follows:

.. code-block:: shell

    python3 -m pip install -i https://test.pypi.org/simple hip-python~=$rocm_version
    # if you want to install the CUDA Python interoperability package too, run:
    python3 -m pip install -i https://test.pypi.org/simple hip-python-as-cuda~=$rocm_version

For other installation options check `hip-python on GitHub <https://github.com/ROCm/hip-python>`_

Installing the git version
--------------------------

You can also install from the git repository. This way you also get the examples.
Please note that this will install all required dependencies in the current environment.
For step-by-step instructions on setting up a development environment, see :ref:`Development Environment <dev-environment>`.

.. code-block:: bash

    git clone https://github.com/benvanwerkhoven/kernel_tuner.git
    cd kernel_tuner
    curl -sSL https://install.python-poetry.org | python3 -
    poetry install

You can install Kernel Tuner with several optional dependencies.
In this we differentiate between development and runtime dependencies.
The development dependencies are ``test`` and ``docs``, and can be installed by appending e.g. ``--with test,docs``.
The runtime dependencies are:

- `cuda`: install pycuda along with kernel_tuner
- `opencl`: install pycuda along with kernel_tuner
- `hip`: install HIP Python along with kernel_tuner
- `tutorial`: install packages required to run the guides

These can be installed by appending e.g. ``-E cuda -E opencl -E hip``.
If you want to go all-out, use ``--all-extras``.

For example, use:
.. code-block:: bash

    poetry install --with test,docs -E cuda -E opencl

To install Kernel Tuner along with all the packages required for development.


Dependencies for the guides
---------------------------

Some addition Python packages are required to run the Jupyter notebook guides.
These packages are commonly used and chances are that you already have these installed.

However, to install Kernel Tuner along with the dependencies to run the guides,
you could use:

.. code-block:: bash

    pip install kernel_tuner[tutorial,cuda]

Or if you have already installed Kernel Tuner and PyCUDA, just use:

.. code-block:: bash

    pip install jupyter matplotlib pandas
