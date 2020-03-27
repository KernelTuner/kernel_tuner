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
<http://benvanwerkhoven.github.io/kernel_tuner/design.html>`__, or discuss it in the issue regarding your additions.

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
are not installed/present. The same holds for tests that require PyOpenCL.

Contributions you make to the Kernel Tuner should not break any of the tests
even if you cannot run them locally.

The examples can be seen as *integration tests* for the Kernel Tuner. Note that
these will also use the installed package.

Building documentation
----------------------
Documentation is located in the ``doc/`` directory. This is where you can type
``make html`` to generate the html pages in the ``doc/build/html`` directory.

The source files used for building the documentation are located in
``doc/source``. The tutorials should be included in the ``tutorials/`` directory
and a symlink can be used to add them to the source file directory before building
documentation.

To update the documentation pages hosted on the GitHub the generated contents of
``doc/build/html`` should be copied to the top-level directory of the
``gh-pages`` branch.
