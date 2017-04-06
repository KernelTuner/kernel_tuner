Installation Guide
==================

The Kernel Tuner requires several packages to be installed. First of all you need a 
working Python version, several Python packages, and optionally CUDA and/or OpenCL 
installations. All of this is explained in detail in this guide.


Python
------

First of all you need a Python installation. I recommend using Python 3 and 
installing it with `Miniconda <https://conda.io/miniconda.html>`__.

Linux users could type the following to download and install Python 3 using Miniconda:

.. code-block:: bash

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

You are of course also free to use your own Python installation, and the Kernel Tuner
is developed to be fully compatible with Python 3.5 and newer, and also Python 2.7.

Installing Python Packages
--------------------------

Note that when you are using a native Python installation, the `pip` commands used to 
install dependencies probably require `sudo` rights. 

Sudo rights are typically not required when using Miniconda or virtual environments.

The following command will install all required dependencies:

.. code-block:: bash

    pip install -r requirements.txt

There are also optional dependencies, explained below.

CUDA and PyCUDA
---------------

Installing CUDA and PyCUDA is optional, because you may want to only use the Kernel 
Tuner for tuning OpenCL or C kernels.

If you want to use the Kernel Tuner to tune 
CUDA kernels you will first need to install the CUDA toolkit 
(https://developer.nvidia.com/cuda-toolkit).

It's very important that you install the CUDA toolkit before trying to install PyCuda.

You can install PyCuda using:

.. code-block:: bash

    pip install pycuda

If you run into trouble with installing PyCuda, make sure you have CUDA installed first.
Also make sure that the Python package Numpy is already installed (this should be the case
because it is also a requirement for the Kernel Tuner).

If you retry the ``pip install pycuda`` command you may need to use the 
``--no-cache-dir`` option to ensure the pycuda installation really starts over.

If this fails, I recommend to see the PyCuda 
installation guide (https://wiki.tiker.net/PyCuda/Installation)


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

After you've installed your OpenCL compiler of choice you can install PyOpenCL using:

.. code-block:: bash

    pip install pyopencl

If this fails, please see the PyOpenCL installation guide (https://wiki.tiker.net/PyOpenCL/Installation)


Installing the Kernel Tuner
---------------------------

So far we've talked about all the dependencies, but not the Kernel Tuner itself.

The easiest way to install is using pip:

.. code-block:: bash

    pip install kernel_tuner

But you can also install from the git repository. This way you also get the 
examples and the tutorials.

.. code-block:: bash

    git clone https://github.com/benvanwerkhoven/kernel_tuner.git
    cd kernel_tuner
    pip install -r requirements.txt
    pip install .

Then go to any of the ``examples/cuda`` or ``examples/opencl`` directories
and see if you can run the ``vector_add.py`` example to test your installation.





