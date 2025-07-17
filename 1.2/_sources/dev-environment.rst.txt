.. toctree::
   :maxdepth: 2

.. role:: bash(code)
   :language: bash

.. _dev-environment:

Development environment
=======================

The following steps help you set up a full development environment. **These steps are only needed for core developers of Kernel Tuner who need to test against multiple Python versions 
or change dependencies of Kernel Tuner.**

For small changes to the code, please see the simplified instructions in the :ref:`simple-dev-env`.

Local setup
^^^^^^^^^^^
Steps with :bash:`sudo` access (e.g. on a local device):

#. Clone the git repository to the desired location: :bash:`git clone https://github.com/KernelTuner/kernel_tuner.git`, and :bash:`cd` to it.
#. Prepare your system for building Python versions.
    * On Ubuntu, run :bash:`sudo apt update && sudo apt upgrade`, and :bash:`sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git`.
#. Install `pyenv <https://github.com/pyenv/pyenv#installation>`__:
    * On Linux, run :bash:`curl https://pyenv.run | bash` (remember to add the output to :bash:`.bash_profile` and :bash:`.bashrc` as specified).
    * On macOS, run :bash:`brew update && brew install pyenv`.
    * After installation, restart your shell. 
#. Install the required Python versions: 
    * On some systems, additional packages may be needed to build Python versions. For example on Ubuntu: :bash:`sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev lzma`.
    * Install the Python versions with: :bash:`pyenv install 3.9 3.10 3.11 3.12 3.13`. The reason we're installing all these versions as opposed to just one, is so we can test against all supported Python versions.
#. Set the Python versions so they can be found: :bash:`pyenv local 3.9 3.10 3.11 3.12 3.13` (replace :bash:`local` with :bash:`global` when not using the virtualenv).
#. Setup a local virtual environment in the folder: :bash:`pyenv virtualenv 3.11 kerneltuner` (or whatever environment name and Python version you prefer).
#. `Install Poetry <https://python-poetry.org/docs/#installing-with-the-official-installer>`__. 
    * Use :bash:`curl -sSL https://install.python-poetry.org | python3 -` to install Poetry.
    * Make sure to add Poetry to :bash:`PATH` as instructed at the end of the installation.
    * Add the poetry export plugin with :bash:`poetry self add poetry-plugin-export`. 
#. Make sure that non-Python dependencies are installed if applicable, such as CUDA, OpenCL or HIP. This is described in :ref:`Installation <installation>`.
#. Apply changes:
    * Re-open the shell for changes to take effect. 
    * Activate the environment with :bash:`pyenv activate kerneltuner`.
    * Make sure :bash:`which python` and :bash:`which pip` point to the expected Python location and version. 
    * Update Pip with :bash:`pip install --upgrade pip`.
#. Install the project, dependencies and extras: :bash:`poetry install --with test,docs -E cuda -E opencl -E hip`, leaving out :bash:`-E cuda`, :bash:`-E opencl` or :bash:`-E hip` if this does not apply on your system. To go all-out, use :bash:`--all-extras`
    * Depending on the environment, it may be necessary or convenient to install extra packages such as :bash:`cupy-cuda11x` / :bash:`cupy-cuda12x`, and :bash:`cuda-python`. These are currently not defined as dependencies for kernel-tuner, but can be part of tests.
    * Do not forget to make sure the paths are set correctly. If you're using CUDA, the desired CUDA version should be in :bash:`$PATH`, :bash:`$LD_LIBARY_PATH` and :bash:`$CPATH`.
    * Re-open the shell for changes to take effect.
#. Check if the environment is setup correctly by running :bash:`pytest` and :bash:`nox`. All tests should pass, except if one or more extras has been left out in the previous step, then these tests will skip gracefully.
    * [Note]: sometimes, changing the NVIDIA driver privileges is required to read program counters and energy measurements. Check if :bash:`cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly` is set to 1. If so, `follow these steps <https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters>`__


Cluster setup
^^^^^^^^^^^^^
Steps without :bash:`sudo` access (e.g. on a cluster):

#. Clone the git repository to the desired location: :bash:`git clone https://github.com/KernelTuner/kernel_tuner.git`.
#. Install Conda with `Mamba <https://mamba.readthedocs.io/en/latest/mamba-installation.html>`__ (for better performance) or `Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`__ (for traditional minimal Conda).
    * [Optional] if you are under quotas or are otherwise restricted by disk space, you can instruct Conda to use a different directory for saving environments by adding the following to your :bash:`.condarc` file:
        .. code-block:: bash

            envs_dirs:
             - /path/to/directory
    * [Optional] both Mamba and Miniconda can be automatically activated via :bash:`~/.bashrc`. Do not forget to add these (usually provided at the end of the installation).
    * Exit the shell and re-enter to make sure Conda is available. :bash:`cd` to the kernel tuner directory.
    * [Optional] if you have limited user folder space, the Pip cache can be pointed elsewhere with the environment variable :bash:`PIP_CACHE_DIR`. The cache location can be checked with :bash:`pip cache dir`. On Linux, to point the entire :bash:`~/.cache` default elsewhere, use the :bash:`XDG_CACHE_HOME` environment variable. 
    * [Optional] update Conda if available before continuing: :bash:`conda update -n base -c conda-forge conda`.
#. Setup a virtual environment: :bash:`conda create --name kerneltuner python=3.11` (or whatever Python version and environment name you prefer).
#. Activate the virtual environment: :bash:`conda activate kerneltuner`.
    * [Optional] to use the correct environment by default, execute :bash:`conda config --set auto_activate_base false`, and add `conda activate kerneltuner` to your :bash:`.bash_profile` or :bash:`.bashrc`.
#. Make sure that non-Python dependencies are loaded if applicable, such as CUDA, OpenCL or HIP. On most clusters it is possible to load (or unload) modules (e.g. CUDA, OpenCL / ROCM). For more information, see :ref:`Installation <installation>`.
    * Do not forget to make sure the paths are set correctly. If you're using CUDA, the desired CUDA version should be in :bash:`$PATH`, :bash:`$LD_LIBARY_PATH` and :bash:`$CPATH`.
    * [Optional] the loading of modules and setting of paths is likely convenient to put in your :bash:`.bash_profile` or :bash:`.bashrc`.
#. `Install Poetry <https://python-poetry.org/docs/#installing-with-the-official-installer>`__. 
    * Use :bash:`curl -sSL https://install.python-poetry.org | python3 -` to install Poetry.
    * Add the poetry export plugin with :bash:`poetry self add poetry-plugin-export`. 
#. Install the project, dependencies and extras: :bash:`poetry install --with test,docs -E cuda -E opencl -E hip`, leaving out :bash:`-E cuda`, :bash:`-E opencl` or :bash:`-E hip` if this does not apply on your system. To go all-out, use :bash:`--all-extras`.
    * If you run into "keyring" or other seemingly weird issues, this is a known issue with Poetry on some systems. Do: :bash:`pip install keyring`, :bash:`python3 -m keyring --disable`.
    * Depending on the environment, it may be necessary or convenient to install extra packages such as :bash:`cupy-cuda11x` / :bash:`cupy-cuda12x`, and :bash:`cuda-python`. These are currently not defined as dependencies for kernel-tuner, but can be part of tests.
    * Verify that your development environment has no missing installs or updates with :bash:`poetry install --sync --dry-run --with test`. 
#. Check if the environment is setup correctly by running :bash:`pytest`. All tests should pass, except if you're not on a GPU node, or one or more extras has been left out in the previous step, then these tests will skip gracefully.
#. Set Nox to use the correct backend and location:
    * Run :bash:`conda -- create-settings-file` to automatically create a settings file. 
    * In this settings file :bash:`noxsettings.toml`, change the :bash:`venvbackend`:
        * If you used Mamba in step 2, to :bash:`mamba`.
        * If you used Miniconda or Anaconda in step 2, to :bash:`conda`.
        * If you used Venv in step 2, to :bash:`venv`.
        * If you used Virtualenv in step 2, this is already the default.
    * Be sure to adjust this when changing backends.
    * The settings file also has :bash:`envdir`, which allows you to `change the directory Nox caches environments in <https://nox.thea.codes/en/stable/usage.html#opt-envdir>`_, particularly helpful if you have a diskquota on your user directory. 
#. [Optional] Run the tests on Nox as described below.


Running tests
^^^^^^^^^^^^^
To run the tests you can use :bash:`nox` (to run against all supported Python versions in isolated environments) and :bash:`pytest` (to run against the local Python version, see below) in the top-level directory.
For full coverage, make Nox use the additional tests (such as cupy and cuda-python) with :bash:`nox -- additional-tests`.

The Nox isolated environments can take up to 1 gigabyte in size, so users tight on diskspace can run :bash:`nox` with the :bash:`small-disk` option. This removes the other environment caches before each session is ran (note that this will take longer to run). A better option would be to change the location environments are stored in with :bash:`envdir` in the :bash:`noxsettings.toml` file. 

Please note that the command-line options can be combined, e.g. :bash:`nox -- additional-tests skip-hip small-disk`. 
If you do not have fully compatible hardware or environment, you can use the following options:

* :bash:`nox -- skip-cuda` to skip tests involving CUDA.
* :bash:`nox -- skip-hip` to skip tests involving HIP.
* :bash:`nox -- skip-opencl` to skip tests involving OpenCL.
* :bash:`nox -- skip-gpu` to skip all tests on the GPU (the same as :bash:`nox -- skip-cuda skip-hip skip-opencl`), especially helpful if you don't have a GPU locally. 

Contributions you make to the Kernel Tuner should not break any of the tests even if you cannot run them locally!

Running with :bash:`pytest` will test against your local Python version and PIP packages. 
In this case, tests that require PyCuda and/or a CUDA capable GPU will be skipped automatically if these are not installed/present. 
The same holds for tests that require PyOpenCL, Cupy, and CUDA.
It is also possible to invoke PyTest from the 'Testing' tab in Visual Studio Code to visualize the testing in your IDE.

The examples can be seen as *integration tests* for the Kernel Tuner.
Note that these will also use the installed package.

Building documentation
^^^^^^^^^^^^^^^^^^^^^^
Documentation is located in the ``doc/`` directory. This is where you can type
``make html`` to generate the html pages in the ``doc/build/html`` directory.
The source files used for building the documentation are located in
``doc/source``.
To locally inspect the documentation before committing you can browse through
the documentation pages generated locally in ``doc/build/html``.

To make sure you have all the dependencies required to build the documentation, at least those in ``--with docs``.
Pandoc is also required, you can install pandoc on Ubuntu using ``sudo apt install pandoc`` and on Mac using ``brew install pandoc``.
For different setups please see `pandoc's install documentation <https://pandoc.org/installing.html>`__.

The documentation pages hosted online are built automatically using GitHub actions.
The documentation pages corresponding to the master branch are hosted in /latest/.
The documentation of the last release is in /stable/. When a new release
is published the documentation for that release will be stored in a directory
created for that release and /stable/ will be updated to point to the last
release. This process is again fully automated using GitHub actions.
