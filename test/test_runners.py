import os
import time

import numpy as np
import pytest

from kernel_tuner import core, tune_kernel, util
from kernel_tuner.interface import Options, _device_options, _kernel_options, _tuning_options
from kernel_tuner.runners.sequential import SequentialRunner

from .context import skip_if_no_pycuda

cache_filename = os.path.dirname(
    os.path.realpath(__file__)) + "/test_cache_file.json"


@pytest.fixture
def env():
    kernel_string = """
    extern "C" __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 100
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    tune_params = dict()
    tune_params["block_size_x"] = [128 + 64 * i for i in range(15)]

    return ["vector_add", kernel_string, size, args, tune_params]


@skip_if_no_pycuda
def test_sequential_runner_alt_block_size_names(env):

    kernel_string = """__global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_dim_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    tune_params = {
        "block_dim_x": [128 + 64 * i for i in range(5)],
        "block_size_y": [1],
        "block_size_z": [1]
    }

    env[1] = kernel_string
    env[-1] = tune_params

    ref = (env[3][1] + env[3][2]).astype(np.float32)
    answer = [ref, None, None, None]

    block_size_names = ["block_dim_x"]

    result, _ = tune_kernel(*env,
                            grid_div_x=["block_dim_x"],
                            answer=answer,
                            block_size_names=block_size_names, objective='time', objective_higher_is_better=False)

    assert len(result) == len(tune_params["block_dim_x"])


@skip_if_no_pycuda
def test_smem_args(env):
    result, _ = tune_kernel(*env,
                            smem_args=dict(size="block_size_x*4"),
                            verbose=True)
    tune_params = env[-1]
    assert len(result) == len(tune_params["block_size_x"])
    result, _ = tune_kernel(
        *env,
        smem_args=dict(size=lambda p: p['block_size_x'] * 4),
        verbose=True)
    tune_params = env[-1]
    assert len(result) == len(tune_params["block_size_x"])


@skip_if_no_pycuda
def test_build_cache(env):
    if not os.path.isfile(cache_filename):
        result, _ = tune_kernel(*env,
                                cache=cache_filename,
                                verbose=False,
                                quiet=True)
        tune_params = env[-1]
        assert len(result) == len(tune_params["block_size_x"])


def test_simulation_runner(env):
    kernel_name, kernel_string, size, args, tune_params = env
    start = time.perf_counter()
    result, res_env = tune_kernel(*env,
                                  cache=cache_filename,
                                  strategy="random_sample",
                                  simulation_mode=True,
                                  strategy_options=dict(fraction=1))
    actual_time = (time.perf_counter() - start) * 1e3  # ms
    assert len(result) == len(tune_params["block_size_x"])

    timings = [
        'total_framework_time', 'total_strategy_time', 'total_compile_time',
        'total_benchmark_time', 'overhead_time'
    ]

    # ensure all keys are there and non zero
    assert all(key in res_env for key in timings)
    assert all(res_env[key] > 0.0 for key in timings)

    # ensure simulation mode and simulated time are properly recorded
    assert "simulated_time" in res_env
    assert "simulation" in res_env and res_env["simulation"]

    # ensure recorded time is sensible number
    recorded_time_including_simulation = sum(res_env[key] for key in timings)
    assert recorded_time_including_simulation - res_env['simulated_time'] > 0

    # ensure difference between recorded time and actual time + simulated less then 10ms
    max_time = actual_time + res_env['simulated_time']
    assert max_time - recorded_time_including_simulation < 10


def test_diff_evo(env):
    result, _ = tune_kernel(*env,
                            strategy="diff_evo",
                            strategy_options=dict(popsize=5),
                            verbose=True,
                            cache=cache_filename,
                            simulation_mode=True)
    assert len(result) > 0


@skip_if_no_pycuda
def test_time_keeping(env):
    kernel_name, kernel_string, size, args, tune_params = env
    answer = [args[1] + args[2], None, None, None]

    options = dict(method="uniform",
                   popsize=10,
                   maxiter=1,
                   mutation_chance=1,
                   max_fevals=10)
    start = time.perf_counter()
    result, env = tune_kernel(*env,
                              strategy="genetic_algorithm",
                              strategy_options=options,
                              verbose=True,
                              answer=answer)
    max_time = (time.perf_counter() - start) * 1e3  # ms

    assert len(result) >= 10

    timings = [
        'total_framework_time', 'total_strategy_time', 'total_compile_time',
        'total_verification_time', 'total_benchmark_time', 'overhead_time'
    ]

    # ensure all keys are there and non zero
    assert all(key in env for key in timings)
    assert all(env[key] > 0.0 for key in timings)

    # check if it all adds up
    recorded_time_spent_tuning = sum(env[key] for key in timings)
    assert 0 < recorded_time_spent_tuning < max_time

    # maximum of 10ms difference between recorded time and actual wallclock time waiting on tune_kernel
    assert max_time - recorded_time_spent_tuning < 10


def test_bayesian_optimization(env):
    for method in [
            "poi", "ei", "lcb", "lcb-srinivas", "multi", "multi-advanced",
            "multi-fast"
    ]:
        print(method, flush=True)
        options = dict(popsize=5, max_fevals=10, method=method)
        result, _ = tune_kernel(*env,
                                strategy="bayes_opt",
                                strategy_options=options,
                                verbose=True,
                                cache=cache_filename,
                                simulation_mode=True)
        assert len(result) > 0


def test_random_sample(env):
    result, _ = tune_kernel(*env,
                            strategy="random_sample",
                            strategy_options={"fraction": 0.1},
                            cache=cache_filename,
                            simulation_mode=True)
    # check that number of benchmarked kernels is 10% (rounded up)
    assert len(result) == 2
    # check all returned results make sense
    for v in result:
        assert v['time'] > 0.0 and v['time'] < 1.0


@skip_if_no_pycuda
def test_interface_handles_compile_failures(env):
    kernel_name, kernel_string, size, args, tune_params = env

    kernel_string = """
    __global__
    void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        #if block_size_x == 256
        // request ridiculously large amount of shared memory to trigger compilation failure
        __shared__ double shared_a[1024*1024];
        #endif

        if (i<n) {
            #if block_size_x == 256
                shared_a[i*1024*1024] = a[i];
                c[i] = shared_a[i*1024] + b[i];
            #else
                c[i] = a[i] + b[i];
            #endif
        }
    }
    """

    results, env = tune_kernel(kernel_name,
                               kernel_string,
                               size,
                               args,
                               tune_params,
                               verbose=True)

    failed_config = [
        record for record in results if record["block_size_x"] == 256
    ][0]
    assert isinstance(failed_config["time"], util.CompilationFailedConfig)


@skip_if_no_pycuda
def test_runner(env):

    kernel_name, kernel_source, problem_size, arguments, tune_params = env

    # create KernelSource
    kernelsource = core.KernelSource(kernel_name,
                                     kernel_source,
                                     lang=None,
                                     defines=None)

    # create option bags
    device = 0
    atol = 1e-6
    platform = 0
    iterations = 7
    verbose = False
    objective = "GFLOP/s"
    metrics = dict({objective: lambda p: 1})
    opts = locals()
    kernel_options = Options([(k, opts.get(k, None))
                              for k in _kernel_options.keys()])
    tuning_options = Options([(k, opts.get(k, None))
                              for k in _tuning_options.keys()])
    device_options = Options([(k, opts.get(k, None))
                              for k in _device_options.keys()])
    tuning_options.cachefile = None

    # create runner
    runner = SequentialRunner(kernelsource,
                              kernel_options,
                              device_options,
                              iterations,
                              observers=None)
    runner.warmed_up = True  # disable warm up for this test

    # select a config to run
    searchspace = []

    # insert configurations to run with this runner in this list
    # each configuration is described as a list of values, one for each tunable parameter
    # the order should correspond to the order of parameters specified in tune_params
    searchspace.append(
        [32])  # vector_add only has one tunable parameter (block_size_x)

    # call the runner
    results = runner.run(searchspace, tuning_options)

    assert len(results) == 1
    assert results[0]['block_size_x'] == 32
    assert len(results[0]['times']) == iterations
