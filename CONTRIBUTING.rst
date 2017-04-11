Contribution guide
==================

The kernel tuner follows the Google Python style guide, with Sphinxdoc 
docstrings for module public functions. If you want to contribute to the project 
please fork it, create a branch including your changes and additions, and create 
a pull request.

Before creating a pull request please ensure the following:

* You have written unit tests to test your additions and all unit tests pass
* The examples still work and produce the same (or better) results
* The code is compatible with both Python 2.7 and Python 3.5
* An entry about the change or addition is created in CHANGELOG.md
* Any matching entries in the roadmap.md are updated/removed

If you are in doubt on where to put your additions to the Kernel Tuner, please 
have look at the `design documentation 
<http://benvanwerkhoven.github.io/kernel_tuner/design.html>`__.

Development setup
-----------------

You can install the packages required to run the tests using:

.. code-block:: bash

    pip install -r requirements-dev.txt

After this command you should be able to run the tests and build the documentation.
See below on how to do that.

Running tests
-------------

To run the tests you can use ``pytest -v`` in the top-level directory. Note that 
pytest tests against the installed package. To update the installed package 
after you've made changes use:

.. code-block:: bash

    pip install --upgrade .
    pytest -v

Note that tests that require PyCuda and/or a CUDA capable GPU will be skipped if these
are not installed/present. The same holds for thats that require PyOpenCL.

Contributions you make to the Kernel Tuner should not break any of the tests 
even if you can not run them locally.

The examples can be seen as *integration tests* for the Kernel Tuner. Note that 
these will also use the installed package.

Building documentation
----------------------

Documentation is located in the ``doc/`` directory. This is where you can type 
``make html`` to generate the html pages in the ``doc/build/html`` directory.

The source files used for building the documentation is located in 
``doc/source``. The tutorials should be included in the ``tutorials/`` directory 
and a symlink can be used to add them to the source files for building 
documentation.

To update the documentation pages hosted on the GitHub the generated contents of 
``doc/build/html`` should be copied to the top-level directory of the 
``gh-pages`` branch.
