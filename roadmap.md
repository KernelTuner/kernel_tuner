# Roadmap for Kernel Tuner

This roadmap presents an overview of the features we are currently planning to
implement. Please note that this is a living document that will evolve as
priorities grow and shift.

## version 0.4.4

 * Module with helper functions for energy tuning

## version 0.4.X

 * JSON specification for cache files
 * Properly support logging again
 * Remote/parallel tuning on different nodes
 * Readers and writers for common auto-tuning interface
 * More OpenACC/OpenMP examples and tests

## version 0.5.0

 * Object-oriented user-interface for multi-problem tuning


## Wish list

These are the things that we would like to implement, but we currently have no
immediate demand for it. If you are interested in any of these, let us know!

 * Provide API for analysis of tuning results
 * Tuning compiler options in combination with other parameters
 * Example that tunes a kernel using thread block re-indexing
 * Extend Fortran support, no more warnings on data types or missing block size parameter etc.
 * Turn the C backend into a more general compiler backend
 * A get_parameterized_kernel_source function to return the parameterized kernel source for inspection
 * Function to generate wrapper kernels for directly calling/testing device functions

