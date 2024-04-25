.. _contributing:

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

If you are in doubt on where to put your additions to the Kernel Tuner, please
have look at the :ref:`design documentation <design documentation>`, or discuss it in the issue regarding your additions.

.. _simple-dev-env:

Simple development setup
------------------------

For small changes to the code you can setup a quick development environment with the following steps:

* :bash:`git clone git@github.com:KernelTuner/kernel_tuner.git`
* :bash:`cd kernel_tuner`
* :bash:`pip install -e .`

To run the tests in your local Python environment:

* :bash:`pip install -r doc/requirements_test.txt`
* :bash:`pytest -v test`

To build the documentation locally:

* :bash:`pip install -r doc/requirements.txt`
* :bash:`cd doc`
* :bash:`make html`

These instructions should be enough for most small contributions. 
For larger changes, or when you need to change the dependencies of Kernel Tuner, please see the documentation on setting up a full development environment.

