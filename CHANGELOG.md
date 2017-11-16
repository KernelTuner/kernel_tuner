# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

### Changed
- bugfix for when using iterations smaller than 3
- the install procedure now uses extras, e.g. [cuda,opencl]

### Added
- checking correct types on argument list and answers

## [0.1.7] - 2017-10-11
### Changed
- bugfix install when scipy not present
- bugfix for GPU cleanup when using Noodles runner
- reworked the way strings are handled internally

### Added
- option to set compiler name, when using C backend

## [0.1.6] - 2017-08-17
### Changed
- actively freeing GPU memory after tuning
- bugfix for 3D grids when using OpenCL

### Added
- support for dynamic parallelism when using PyCUDA
- option to use differential evolution optimization
- global optimization strategies basinhopping, minimize

## [0.1.5] - 2017-07-21
### Changed
- option to pass a fraction to the sample runner
- fixed a bug in memset for OpenCL backend

### Added
- parallel tuning on single node using Noodles runner
- option to pass new defaults for block dimensions
- option to pass a Python function as code generator
- option to pass custom function for output verification

## [0.1.4] - 2017-06-14
### Changed
- device and kernel name are printed by runner
- tune_kernel also returns a dict with environment info
- using different timer in C vector add example

## [0.1.3] - 2017-04-06
### Changed
- changed how scalar arguments are handled internally

### Added
- separate install and contribution guides

## [0.1.2] - 2017-03-29
### Changed
- allow non-tuple problem_size for 1D grids
- changed default for grid_div_y from None to block_size_y
- converted the tutorial to a Jupyter Notebook
- CUDA backend prints device in use, similar to OpenCL backend
- migrating from nosetests to pytest
- rewrote many of the examples to save results to json files

### Added
- full support for 3D grids, including option for grid_div_z
- separable convolution example

## [0.1.1] - 2017-02-10
### Changed
- changed the output format to list of dictionaries

### Added
- option to set compiler options

## [0.1.0] - 2016-11-02
### Changed
- verbose now also prints debug output when correctness check fails
- restructured the utility functions into util and core
- restructured the code to prepare for different strategies
- shortened the output printed by the tune_kernel
- allowing numpy integers for specifying problem size

### Added
- a public roadmap
- requirements.txt
- example showing GPU code unit testing with the Kernel Tuner
- support for passing a (list of) filenames instead of kernel string
- runner that takes a random sample of 10 percent
- support for OpenCL platform selection
- support for using tuning parameter names in the problem size

## [0.0.1] - 2016-06-14
### Added
- A function to type check the arguments to the kernel
- Example (convolution) that tunes the number of streams 
- Device interface to C functions, for tuning host code
- Correctness checks for kernels during tuning
- Function for running a single kernel instance
- CHANGELOG file
- Compute Cartesian product and process restrictions before main loop
- Python 3.5 compatible code, thanks to Berend
- Support for constant memory arguments to CUDA kernels
- Use of mocking in unittests
- Reporting coverage to codacy
- OpenCL support
- Documentation pages with Convolution and Matrix Multiply examples
- Inspecting device properties at runtime
- Basic Kernel Tuning functionality


