Installation Guide
==================

The Kernel Tuner requires several packages to be installed. First of all, you need a 
working Python version, several Python packages, and optionally CUDA and/or OpenCL 
installations. All of this is explained in detail in this guide.


Python
------

You need a Python installation. I recommend using Python 3 and 
installing it with `Miniconda <https://conda.io/miniconda.html>`__.

Linux users could type the following to download and install Python 3 using Miniconda:

.. code-block:: bash

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

You are of course also free to use your own Python installation, and the Kernel Tuner
is developed to be fully compatible with Python 3.5 and newer, and also Python 2.7.

Installing Python Packages
--------------------------

Note that when you are using a native Python installation, the `pip` command used 
Kernel Tuner and its dependencies require `sudo` rights for system wide installation. 

Sudo rights are typically not required when using Miniconda or virtual environments.
You could also use e.g. the `--user` or `--prefix` option of `pip` to install into 
your home directory,
this requires that your home directory is on your `$PYTHONPATH` environment variable
(see for further details the pip documentation).

The following command will install Kernel Tuner together with the required dependencies:

.. code-block:: bash

    pip install kernel_tuner

There are also optional dependencies, explained below.

CUDA and PyCUDA
---------------

Installing CUDA and PyCUDA is optional, because you may want to only use Kernel 
Tuner for tuning OpenCL or C kernels. 

If you want to use the Kernel Tuner to tune 
CUDA kernels you will first need to install the CUDA toolkit 
(https://developer.nvidia.com/cuda-toolkit). A recent version of the 
CUDA toolkit (and the PyCUDA Python bindings for CUDA) are 
recommended (older version may work, but may not support all features of 
the Kernel Tuner). 

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


OpenCL and PyOpenCL
-------------------

Before we can install PyOpenCL you'll need an OpenCL compiler. There are several 
OpenCL compilers available depending on the OpenCL platform you want to your 
code to run on.

* `AMD APP SDK <http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/>`__
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


Installing the Kernel Tuner
---------------------------

You can also install from the git repository. This way you also get the 
examples and the tutorials.

.. code-block:: bash

    git clone https://github.com/benvanwerkhoven/kernel_tuner.git
    cd kernel_tuner
    pip install .

You can install Kernel Tuner with several optional dependencies, the full list is:

- `cuda`: install pycuda along with kernel_tuner
- `opencl`: install pycuda along with kernel_tuner
- `doc`: installs packages required to build the documentation
- `tutorial`: install packages required to run the tutorials
- `dev`: install everything you need to start development on Kernel Tuner

For example, use:
```
pip install .[dev,cuda,opencl]
```
To install Kernel Tuner along with all the packages required for development.


Dependencies for the Tutorial
-----------------------------

Some addition Python packages are required to run the tutorial. These packages are
actually very commonly used and chances are that you already have these installed.

However, to install Kernel Tuner along with the dependencies to run the tutorials,
you could use:

.. code-block:: bash

    pip install kernel_tuner[tutorial,cuda]

Or if you have already installed Kernel Tuner and PyCUDA, just use:

.. code-block:: bash

    pip install jupyter matplotlib pandas




