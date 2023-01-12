Contribution guide
==================
Thank you for considering to contribute to Kernel Tuner!

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

Kernel Tuner follows the Google Python style guide, with Sphinxdoc docstrings for module public functions. Please use `pylint` to check your Python changes.

Before creating a pull request please ensure the following:

* You have written unit tests to test your additions and all unit tests pass
* The examples still work and produce the same (or better) results
* The code is compatible with Python 3.5 or newer
* You have run `pylint` to check your code
* An entry about the change or addition is created in CHANGELOG.md
* Any matching entries in the roadmap.md are updated/removed

If you are in doubt on where to put your additions to the Kernel Tuner, please
have look at the `design documentation
<https://kerneltuner.github.io/kernel_tuner/stable/design.html>`__, or discuss it in the issue regarding your additions.

Development setup
-----------------
You can install the packages required to run the tests using:

.. code-block:: bash

    pip install -e .[dev]

After this command you should be able to run the tests and build the documentation.
See below on how to do that. The ``-e`` flag installs the package in *development mode*.
This means files are not copied, but linked to, such that your installation tracks
changes in the source files.

Running tests
-------------
To run the tests you can use ``pytest -v test/`` in the top-level directory.

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

To make sure you have all the dependencies required to build the documentation,
you can install the extras using ``pip install -e .[doc]``. Pandoc is also required,
you can install pandoc on ubuntu using ``sudo apt install pandoc``, for different
setups please see `pandoc's install documentation <https://pandoc.org/installing.html>`__.

The documentation pages hosted online are built automatically using GitHub actions.
The documentation pages corresponding to the master branch are hosted in /latest/.
The documentation of the last release is in /stable/. When a new release
is published the documentation for that release will be stored in a directory
created for that release and /stable/ will be updated to point to the last
release. This process is again fully automated using GitHub actions.
