# Roadmap for the Kernel Tuner

This roadmap presents an overview of the features we are currently planning to
implement. Please note that this is a living document that will evolve as
priorities grow and shift.

### version 0.3.0

This is the list of features that we want to have implemented by the next version.

 * Enable setting search strategy parameters through the user interface
 * Extend Fortran support, no more warnings on data types or missing block size parameter etc.
 * Turn the C backend into a more general compiler backend
 * A get_parameterized_kernel_source function to return the parameterized kernel source for inspection
 * A test_kernel function to perform parameterized testing without tuning
 * Function to instrument source files with parameter values after tuning
 * Function to generate wrapper kernels for directly calling device functions
 
### version 1.0.0

These functions are to be implemented by version 1.0.0, but may already be
implemented in earlier versions.

 * Functionality for including auto-tuned kernels in applications
 * Tuning kernels in parallel on a set of nodes in a GPU cluster

### Low priority

These are the things that we would like to implement, but we currently have no
demand for it. If you are interested in any of these, let us know!

 * Option to set dynamically allocated shared memory for CUDA backend
 * Option to set function that computes search space restriction, instead of a list of strings
 * Option to set function that computes grid dimensions instead of grid divisor lists
 * Provide API for analysis of tuning results
 * Tuning compiler options in combination with other parameters
 * Example that tunes a kernel using thread block re-indexing
 * Example CUDA host code that uses runtime compilation


