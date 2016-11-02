# Roadmap for the Kernel Tuner

This roadmap presents an overview of the features we are currently planning to
implement. Please note that this is a living document that will evolve as
priorities grow and shift.

### version 0.2.0

This is the list of features that we want to have implemented by the next version.

 * Option to store tuning results in a file (e.g. json, csv, ... )
 * Option to set a function that performs output verfication, instead of numpy.allclose()
 * Option to change defaults for 'block_size_x', and so on
 * Option to set a function that computes search space restriction, instead of a list of strings
 * Option to set compiler name, when using C backend
 * Option to set compiler options

### version 1.0.0

These functions are to be implemented by version 1.0.0, but may already be
implemented in earlier versions.

 * Tuning kernels in parallel on a single node
 * Tuning kernels in parallel on a set of nodes in a GPU clusters
 * Tuning kernels using machine learning or search strategies
 * Store tuning results in a database and provide an API for analysis

### Low priority

These are the things that we would like to implement, but we currently have no
demand for it. If you are interesting in any of these, let us know!

 * Tuning compiler options in combination other parameters kernel
 * Example that tunes a kernel using thread block re-indexing
 * Example host code that runs a pipeline of kernels
 * Example CUDA host code that uses runtime compilation


