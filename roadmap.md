# Roadmap for the Kernel Tuner

This roadmap presents an overview of the features we are currently planning to
implement. Please note that this is a living document that will evolve as
priorities grow and shift.

## version 0.4.0

 * Support for C++ templated kernels through NVRTC and template parameter tuning
 * Document a vocabulary of reserved/special tunable parameter names

## version 0.5.0

 * Allow strategies to tune for a metric other than time

## version 0.9.0

 * Multi-objective optimization

## version 1.0.0

These functions are to be implemented by version 1.0.0, but may already be
implemented in earlier versions.

 * Tuning kernels in parallel on a set of nodes in a GPU cluster
 * Functionality for including auto-tuned kernels in applications

## Wish list

These are the things that we would like to implement, but we currently have no
immediate demand for it. If you are interested in any of these, let us know!

 * Provide API for analysis of tuning results
 * Tuning compiler options in combination with other parameters
 * Example that tunes a kernel using thread block re-indexing
 * Example CUDA host code that uses runtime compilation
 * Extend Fortran support, no more warnings on data types or missing block size parameter etc.
 * Turn the C backend into a more general compiler backend
 * A get_parameterized_kernel_source function to return the parameterized kernel source for inspection
 * Function to generate wrapper kernels for directly calling/testing device functions
 * Function to instrument source files with parameter values after tuning
