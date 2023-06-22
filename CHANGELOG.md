# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

## [0.4.5] - 2023-06-01
### Added
- PMTObserver to measure power and energy on various platforms

### Changed
- Improved functionality for storing output and metadata files
- Updated PowerSensorObserver to support PowerSensor3
- Refactored interal interfaces of runners and backends
- Bugfix in interface to set objective and optimization direction

## [0.4.4] - 2023-03-09
### Added
- Support for using time_limit in simulation mode
- Helper functions for energy tuning
- Example to show ridge frequency and power-frequency model
- Functions to store tuning output and metadata

### Changed
- Changed what timings are stored in cache files
- No longer inserting partial loop unrolling factor of 0 in CUDA

## [0.4.3] - 2022-10-19
### Added
- A new backend that uses Nvidia cuda-python
- Support for locked clocks in NVMLObserver
- Support for measuring core voltages using NVML
- Support for custom preprocessor definitions
- Support for boolean scalar arguments in PyCUDA backend

### Changed
- Migrated from github.com/benvanwerkhoven to github.com/KernelTuner
- Significant update to the documentation pages
- Unified benchmarking loops across backends
- Backends are no longer context managers
- Replaced the method for measuring power consumption using NVML
- Improved NVML measurements of temperature and clock frequencies
- bugfix in parse_restrictions when using and/or in expressions
- bugfix in GreedyILS when using neighbor method "adjacent"
- bugfix in Bayesian Optimization for small problems

## [0.4.2] - 2022-05-23
### Added
- new optimization strategies: dual annealing, greedly ILS, ordered greedy MLS, greedy MLS
- support for constant memory in cupy backend
- constraint solver to cut down time spent in creating search spaces
- support for custom tuning objectives
- support for max_fevals and time_limit in strategy_options of all strategies

### Removed
- alternative Bayesian Optimization strategies that could not be used directly
- C++ wrapper module that was too specific and hardly used

### Changed
- string-based restrictions are compiled into functions for improved performance
- genetic algorithm, MLS, ILS, random, and simulated annealing use new search space object
- diff evo, firefly, PSO are initialized using population of all valid configurations
- all strategies except brute_force strictly adhere to max_fevals and time_limit
- simulated annealing adapts annealing schedule to max_fevals if supplied
- minimize, basinhopping, and dual annealing start from a random valid config

## [0.4.1] - 2021-09-10
### Added
- support for PyTorch Tensors as input data type for kernels
- support for smem_args in run_kernel
- support for (lambda) function and string for dynamic shared memory size
- a new Bayesian Optimization strategy

### Changed
- optionally store the kernel_string with store_results
- improved reporting of skipped configurations

## [0.4.0] - 2021-04-09
### Added
- support for (lambda) function instead of list of strings for restrictions
- support for (lambda) function instead of list for specifying grid divisors
- support for (lambda) function instead of tuple for specifying problem_size
- function to store the top tuning results
- function to create header file with device targets from stored results
- support for using tuning results in PythonKernel
- option to control measurements using observers
- support for NVML tunable parameters
- option to simulate auto-tuning searches from existing cache files
- Cupy backend to support C++ templated CUDA kernels
- support for templated CUDA kernels using PyCUDA backend
- documentation on tunable parameter vocabulary

## [0.3.2] - 2020-11-04
### Added
- support loop unrolling using params that start with loop_unroll_factor
- always insert "define kernel_tuner 1" to allow preprocessor ifdef kernel_tuner
- support for user-defined metrics
- support for choosing the optimization starting point x0 for most strategies

### Changed
- more compact output is printed to the terminal
- sequential runner runs first kernel in the parameter space to warm up device
- updated tutorials to demonstrate use of user-defined metrics

## [0.3.1] - 2020-06-11
### Added
- kernelbuilder functionality for including kernels in Python applications
- smem_args option for dynamically allocated shared memory in CUDA kernels

### Changed
- bugfix for Nvidia devices without internal current sensor

## [0.3.0] - 2019-12-20
### Changed
- fix for output checking, custom verify functions are called just once
- benchmarking now returns multiple results not only time
- more sophisticated implementation of genetic algorithm strategy
- how the "method" option is passed, now use strategy_options

### Added
- Bayesian Optimizaton strategy, use strategy="bayes_opt"
- support for kernels that use texture memory in CUDA
- support for measuring energy consumption of CUDA kernels
- option to set strategy_options to pass strategy specific options
- option to cache and restart from tuned kernel configurations cachefile

### Removed
- Python 2 support, it may still work but we no longer test for Python 2
- Noodles parallel runner

## [0.2.0] - 2018-11-16
### Changed
- no longer replacing kernel names with instance strings during tuning
- bugfix in tempfile creation that lead to too many open files error

### Added
- A minimal Fortran example and basic Fortran support
- Particle Swarm Optimization strategy, use strategy="pso" 
- Simulated Annealing strategy, use strategy="simulated_annealing" 
- Firefly Algorithm strategy, use strategy="firefly_algorithm" 
- Genetic Algorithm strategy, use strategy="genetic_algorithm" 

## [0.1.9] - 2018-04-18
### Changed
- bugfix for C backend for byte array arguments
- argument type mismatches throw warning instead of exception

### Added
- wrapper functionality to wrap C++ functions
- citation file and zenodo doi generation for releases

## [0.1.8] - 2017-11-23
### Changed
- bugfix for when using iterations smaller than 3
- the install procedure now uses extras, e.g. [cuda,opencl]
- option quiet makes tune_kernel completely quiet
- extensive updates to documentation

### Added
- type checking for kernel arguments and answers lists
- checks for reserved keywords in tunable paramters
- checks for whether thread block dimensions are specified
- printing units for measured time with CUDA and OpenCL
- option to print all measured execution times

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


