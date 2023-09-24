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

Development environment
-----------------------
The following steps help you set up a minimal development environment. This is just a suggestion, and can be done in many other ways.

Steps with :bash:`sudo` access:

#. Clone the git repository to the desired location
#. Install `pyenv <https://github.com/pyenv/pyenv#installation>`__: :bash:`curl https://pyenv.run | bash` (remember to add the output to :bash:`.bash_profile` and :bash:`.bashrc` as specified)
    * [Optional] setup a local virtual environment in the folder: pyenv virtualenv [virtualenv-name]
#. Install the required Python versions: :bash:`pyenv install 3.8 3.9 3.10 3.11`
#. Set the Python versions so they can be found: :bash:`pyenv global 3.8 3.10 3.11` (replace :bash:`global` with :bash:`local` when using the virtualenv)
#. Install Nox: :bash:`pip install nox`

Steps without :bash:`sudo` access (e.g. on a cluster):

#. Clone the git repository to the desired location
#. Install Conda with `Mamba <https://mamba.readthedocs.io/en/latest/mamba-installation.html>`__ (for better performance) or `Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`__ (for traditional minimal Conda).
    * [Optional] both Mamba and Miniconda can be automatically activated via :bash:`~/.bashrc`. Do not forget to add these (usually mentioned at the end of the installation).
#. Setup a Conda virtual environment for each of the required Python versions: :bash:`conda create --name python3.x python=3.x`. Verify the environments are setup with `conda info --envs`.
#. Install Nox: :bash:`pip install nox`
#. Set Nox to use the correct backend:
    * If you used Mamba in step 2: :bash:`nox --default-venv-backend mamba`
    * If you used Miniconda or Anaconda in step 2: :bash:`nox --default-venv-backend conda`

Contributing Code
-----------------
For contributing code to Kernel Tuner please select an issue to work on or create a new issue to propose a change or addition. For significant changes, it is required to first create an issue and discuss the proposed changes. Then fork the repository, create a branch, one per change or addition, and create a pull request.

Kernel Tuner follows the Google Python style guide, with Sphinxdoc docstrings for module public functions.

Before creating a pull request please ensure the following:

* You have written unit tests to test your additions and all unit tests pass (run :bash:`nox`). If you do not have the required hardware, you can run :bash:`nox -- skip-gpu`, or :bash:`skip-cuda`, :bash:`skip-hip`, :bash:`skip-opencl`.
* The examples still work and produce the same (or better) results
* An entry about the change or addition is created in :bash:`CHANGELOG.md`
* Any matching entries in the roadmap.md are updated/removed

If you are in doubt on where to put your additions to the Kernel Tuner, please
have look at the `design documentation
<https://kerneltuner.github.io/kernel_tuner/stable/design.html>`__, or discuss it in the issue regarding your additions.

Development setup
-----------------
Afer cloning, you can install the packages required to run the tests using:

.. code-block:: bash

    poetry install --no-root --with test,docs
    pip install -e .

After this command you should be able to run the tests and build the documentation.
See below on how to do that. The :bash:`-e` flag installs the package in *development mode*.
This means files are not copied, but linked to, such that your installation tracks
changes in the source files.
Additionally you can install any of the optional runtime dependencies by appending e.g. :bash:`-E cuda`, `-E opencl` to the Poetry command.
If you want to go all-out, use :bash:`--all-extras`.


Running tests
-------------
To run the tests you can use :bash:`pytest` (to run against the local Python version) and :bash:`nox` (to run against all supported Python versions) in the top-level directory.

Note that tests that require PyCuda and/or a CUDA capable GPU will be skipped if these
are not installed/present. The same holds for tests that require PyOpenCL, Cupy, Nvidia CUDA.

Contributions you make to the Kernel Tuner should not break any of the tests
even if you cannot run them locally.

The examples can be seen as *integration tests* for the Kernel Tuner. Note that
these will also use the installed package.

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
