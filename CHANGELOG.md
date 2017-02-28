# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]


## [0.1.1] - 2017-02-10
### Changed
- migrating from nosetests to pytest
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


