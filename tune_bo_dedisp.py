#!/usr/bin/env python
import os
from collections import OrderedDict
from pathlib import Path

import kernel_tuner as kt

nr_dms = 2048
nr_samples = 25000
nr_channels = 1536
max_shift = 650
nr_samples_per_channel = (nr_samples+max_shift)
down_sampling = 1
dm_first = 0.0
dm_step = 0.02

channel_bandwidth = 0.1953125
sampling_time = 0.00004096
min_freq = 1425.0
max_freq = min_freq + (nr_channels-1) * channel_bandwidth


def tune(device_name, strategy="bayes_opt_BOTorch", strategy_options={ 'max_fevals': 1500 }, lang='HIP', verbose=True, quiet=False, simulation_mode=True, profiling=True):

    args = []

    answer = [None, None, None]

    problem_size = (nr_samples, nr_dms, 1)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [1, 2, 4, 8] + [16*i for i in range(1,3)]
    tune_params["block_size_y"] = [8*i for i in range(4,33)]
    tune_params["block_size_z"] = [1]
    tune_params["tile_size_x"] = [i for i in range(1,5)]
    tune_params["tile_size_y"] = [i for i in range(1,9)]
    tune_params["tile_stride_x"] = [0, 1]
    tune_params["tile_stride_y"] = [0, 1]
    tune_params["loop_unroll_factor_channel"] = [0] #+ [i for i in range(1,nr_channels+1) if nr_channels % i == 0] #[i for i in range(nr_channels+1)]

    cp = [f"-I{os.path.dirname(os.path.realpath(__file__))}"]


    check_block_size = "32 <= block_size_x * block_size_y <= 1024"
    check_loop_x = "loop_unroll_factor_x <= tile_size_x and tile_size_x % loop_unroll_factor_x == 0"
    check_loop_y = "loop_unroll_factor_y <= tile_size_y and tile_size_y % loop_unroll_factor_y == 0"
    check_loop_channel = f"loop_unroll_factor_channel <= {nr_channels} and loop_unroll_factor_channel and {nr_channels} % loop_unroll_factor_channel == 0"

    check_tile_stride_x = "tile_size_x > 1 or tile_stride_x == 0"
    check_tile_stride_y = "tile_size_y > 1 or tile_stride_y == 0"

    config_valid = [check_block_size, check_tile_stride_x, check_tile_stride_y]

    metrics = OrderedDict()
    gbytes = (nr_dms * nr_samples * nr_channels)/1e9
    metrics["GB/s"] = lambda p: gbytes / (p['time'] / 1e3)

    directory = Path(__file__).parent / "../autotuning_methodology/cached_data_used/"
    cache_dir = directory / "cachefiles/dedispersion_milo"
    cache_filename = f"{device_name}.json"
    transfer_learning_caches = [p for p in cache_dir.iterdir() if not p.stem.endswith("_T4") and p.name != cache_filename]

    assert directory.exists()
    if lang == "CUDA":
        kernel_file = directory / "kernels/dedisp_milo/dedispersion.cu"
    elif lang == "HIP":
        kernel_file = directory / "kernels/dedisp_milo/dedispersion.cu.hip"
    else:
        raise ValueError(f"Invalid {lang=}")

    def run():
        return kt.tune_kernel("dedispersion_kernel", kernel_file, problem_size, args, tune_params,
                                answer=answer, compiler_options=cp, restrictions=config_valid, device=0,
                                cache=cache_dir / cache_filename, lang=lang, iterations=32, metrics=metrics, 
                                simulation_mode=simulation_mode, verbose=verbose, quiet=quiet, strategy=strategy, 
                                strategy_options=strategy_options, transfer_learning_caches=transfer_learning_caches)
    
    # start tuning
    if profiling:
        import cProfile

        with cProfile.Profile() as pr:
            results, env = run()
            if profiling:
                pr.dump_stats('bo_prof_torchfit_2.prof')
    else:
        results, env = run()

    return results, env

if __name__ == "__main__":

    tune("A100")
