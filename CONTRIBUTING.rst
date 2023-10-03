Contribution guide
==================
Thank you for considering to contribute to Kernel Tuner!

.. role:: bash(code)
   :language: bash

Reporting Issues
----------------
Not all contributions are code, creating an issue also helps us to improve. When you create an issue about a problem, please ensure the following:

* Describe what you expected to happen.
* If possible, include a minimal example to help us reproduce the issue.
* Describe what actually happened, including the output of any errors printed.
* List the version of Python, CUDA or OpenCL, and C compiler, if applicable.

Contributing Code
-----------------
For contributing code to Kernel Tuner please select an issue to work on or create a new issue to propose a change or addition. For significant changes, it is required to first create an issue and discuss the proposed changes. Then fork the repository, create a branch, one per change or addition, and create a pull request.

Kernel Tuner follows the Google Python style guide, with Sphinxdoc docstrings for module public functions.

Before creating a pull request please ensure the following:

* You are working in an up-to-date development environment
* You have written unit tests to test your additions and all unit tests pass (run :bash:`nox`). If you do not have the required hardware, you can run :bash:`nox -- skip-gpu`, or :bash:`skip-cuda`, :bash:`skip-hip`, :bash:`skip-opencl`.
* The examples still work and produce the same (or better) results
* An entry about the change or addition is created in :bash:`CHANGELOG.md`
* Any matching entries in the roadmap.md are updated/removed

If you are in doubt on where to put your additions to the Kernel Tuner, please
have look at the `design documentation
<https://kerneltuner.github.io/kernel_tuner/stable/design.html>`__, or discuss it in the issue regarding your additions.


Development environment
-----------------------
The following steps help you set up a development environment.

Local setup
^^^^^^^^^^^
Steps with :bash:`sudo` access (e.g. on a local device):

#. Clone the git repository to the desired location: :bash:`git clone https://github.com/KernelTuner/kernel_tuner.git`, and :bash:`cd` to it.
#. Install `pyenv <https://github.com/pyenv/pyenv#installation>`__: :bash:`curl https://pyenv.run | bash` (remember to add the output to :bash:`.bash_profile` and :bash:`.bashrc` as specified).
    * [Optional] setup a local virtual environment in the folder: :bash:`pyenv virtualenv kerneltuner` (or whatever environment name you prefer).
#. Install the required Python versions: :bash:`pyenv install 3.8 3.9 3.10 3.11`.
#. Set the Python versions so they can be found: :bash:`pyenv global 3.8 3.10 3.11` (replace :bash:`global` with :bash:`local` when using the virtualenv).
#. `Install Poetry <https://python-poetry.org/docs/#installing-with-the-official-installer>`__: :bash:`curl -sSL https://install.python-poetry.org | python3 -`.
#. Make sure that non-Python dependencies are installed if applicable, such as CUDA, OpenCL or HIP. This is described in `Installation <https://kerneltuner.github.io/kernel_tuner/stable/install.html>`__.
#. Install the project, dependencies and extras: :bash:`poetry install --with test,docs -E cuda -E opencl -E hip`, leaving out :bash:`-E cuda`, :bash:`-E opencl` or :bash:`-E hip` if this does not apply on your system. To go all-out, use :bash:`--all-extras`
    * Depending on the environment, it may be necessary or convenient to install extra packages such as :bash:`cupy-cuda11x` / :bash:`cupy-cuda12x`, and :bash:`cuda-python`. These are currently not defined as dependencies for kernel-tuner, but can be part of tests.
    * Do not forget to make sure the paths are set correctly. If you're using CUDA, the desired CUDA version should be in :bash:`$PATH`, :bash:`$LD_LIBARY_PATH` and :bash:`$CPATH`.
#. Check if the environment is setup correctly by running :bash:`pytest`. All tests should pass, except if one or more extras has been left out in the previous step, then these tests will skip gracefully.


Cluster setup
^^^^^^^^^^^^^
Steps without :bash:`sudo` access (e.g. on a cluster):

#. Clone the git repository to the desired location: :bash:`git clone https://github.com/KernelTuner/kernel_tuner.git`.
#. Install Conda with `Mamba <https://mamba.readthedocs.io/en/latest/mamba-installation.html>`__ (for better performance) or `Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`__ (for traditional minimal Conda).
    * [Optional] both Mamba and Miniconda can be automatically activated via :bash:`~/.bashrc`. Do not forget to add these (usually mentioned at the end of the installation).
    * Exit the shell and re-enter to make sure Conda is available, :bash:`cd` to the kernel tuner directory.
    * [Optional] update Conda if available before continuing: :bash:`conda update -n base -c conda-forge conda`.
#. Setup a virtual environment: :bash:`conda create --name kerneltuner python=3.11` (or whatever Python version and environment name you prefer).
#. Activate the virtual environment: :bash:`conda activate kerneltuner`.
    * [Optional] to use the correct environment by default, execute :bash:`conda config --set auto_activate_base false`, and add `conda activate kerneltuner` to your :bash:`.bash_profile` or :bash:`.bashrc`.
    * Make sure that non-Python dependencies are loaded if applicable, such as CUDA, OpenCL or HIP. On most clusters it is possible to load (or unload) modules (e.g. CUDA, OpenCL / ROCM). For more information, see `Installation <https://kerneltuner.github.io/kernel_tuner/stable/install.html>`__.
    * Do not forget to make sure the paths are set correctly. If you're using CUDA, the desired CUDA version should be in :bash:`$PATH`, :bash:`$LD_LIBARY_PATH` and :bash:`$CPATH`.
    * [Optional] the loading of modules and setting of paths is likely convenient to put in your :bash:`.bash_profile` or :bash:`.bashrc`.
#. `Install Poetry <https://python-poetry.org/docs/#installing-with-the-official-installer>`__: :bash:`curl -sSL https://install.python-poetry.org | python3 -`.
#. Install the project, dependencies and extras: :bash:`poetry install --with test,docs -E cuda -E opencl -E hip`, leaving out :bash:`-E cuda`, :bash:`-E opencl` or :bash:`-E hip` if this does not apply on your system. To go all-out, use :bash:`--all-extras`.
    * If you run into "keyring" or other seemingly weird issues, this is a known issue with Poetry on some systems. Do: :bash:`pip install keyring`, :bash:`python3 -m keyring --disable`.
    * Depending on the environment, it may be necessary or convenient to install extra packages such as :bash:`cupy-cuda11x` / :bash:`cupy-cuda12x`, and :bash:`cuda-python`. These are currently not defined as dependencies for kernel-tuner, but can be part of tests.
#. Check if the environment is setup correctly by running :bash:`pytest`. All tests should pass, except if you're not on a GPU node, or one or more extras has been left out in the previous step, then these tests will skip gracefully.
#. Set Nox to use the correct backend:
    * If you used Mamba in step 2: :bash:`echo "mamba" > noxenv.txt`.
    * If you used Miniconda or Anaconda in step 2: :bash:`echo "conda" > noxenv.txt`.
    * If you alternatively set up with Venv: :bash:`echo "venv" > noxenv.txt`.
    * If you set up with Virtualenv, do not create this file, as this is already the default.
    * Be sure to adjust or remove this file when changing backends.


Running tests
-------------
To run the tests you can use :bash:`nox` (to run against all supported Python versions in isolated environments) and :bash:`pytest` (to run against the local Python version) in the top-level directory.
It's also possible to invoke PyTest from the 'Testing' tab in Visual Studio Code.
The isolated environments can take up to 1 gigabyte in size, so users tight on diskspace can run :bash:`nox` with the :bash:`small-disk` option. This removes the other environment caches before each session is ran.

Note that tests that require PyCuda and/or a CUDA capable GPU will be skipped if these
are not installed/present. The same holds for tests that require PyOpenCL, Cupy, Nvidia CUDA.

Contributions you make to the Kernel Tuner should not break any of the tests even if you cannot run them locally.

The examples can be seen as *integration tests* for the Kernel Tuner.
Note that these will also use the installed package.

Building documentation
----------------------
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
