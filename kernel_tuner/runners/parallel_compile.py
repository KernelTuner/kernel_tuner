"""A runner that compiles configurations in parallel threads, and benchmarks them one after the other."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from kernel_tuner.runners.sequential import SequentialRunner
from kernel_tuner.util import ErrorConfig, Timer

# The maximum number of worker processes used to compile kernels when the number of workers is not specified.
# Each worker process creates its own CUDA context and copy of the kernel arguments on the GPU.
DEFAULT_MAX_BUILD_PROCESSES = 8


class ParallelCompileRunner(SequentialRunner):
    """ParallelCompileRunner compiles configurations in parallel on the local machine.

    The configurations passed to run() are processed in chunks. The kernels of all configurations in a chunk
    are compiled in parallel threads, and once all compilations have finished, the configurations are
    verified and benchmarked one after the other, exactly like the SequentialRunner does. This way, the
    compilation of other kernels does not affect the benchmarks.

    Compilation is done in threads, which only reduces compile time if the compiler does not hold Python's
    global interpreter lock, for example because it runs as a separate process, or in C/C++ code that releases
    the lock. Backends that do not implement DeviceInterface.build_kernel compile the kernels when they are
    loaded, in which case the configurations are compiled one after the other.

    Python DSLs that cannot compile kernels in parallel threads (Numba, Warp, cuTile, CuTe, and Tilus) compile
    the kernels in worker processes instead, from which the main process loads the compiled kernels. Taichi kernels
    are compiled one at a time in the main thread.
    """

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers, num_workers=None):
        """Instantiate the ParallelCompileRunner.

        :param num_workers: The number of threads used for compilation, by default the number of CPU cores.
        :type num_workers: int
        """
        super().__init__(kernel_source, kernel_options, device_options, iterations, observers)
        self.num_workers = num_workers or os.cpu_count() or 1
        # Compiling a few kernels per thread balances the load over the threads, while limiting
        # the number of compiled kernels that are kept in memory until they are benchmarked.
        self.chunk_size = 4 * self.num_workers

        # some backends compile kernels that cannot be compiled in threads in worker processes instead,
        # each worker process uses GPU memory, so the default number of processes is limited
        if hasattr(self.dev.dev, "build_processes"):
            self.dev.dev.build_processes = num_workers or min(self.num_workers, DEFAULT_MAX_BUILD_PROCESSES)

    def shutdown(self):
        """Stop the worker processes that the backend may have started to compile kernels."""
        if hasattr(self.dev.dev, "shutdown_build_processes"):
            self.dev.dev.shutdown_build_processes()

    def run(self, parameter_space, tuning_options):
        """Compile the configurations in parameter_space in parallel, and benchmark them one after the other.

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param tuning_options: A dictionary with all options regarding the tuning process.
        :type tuning_options: kernel_tuner.interface.Options

        :returns: A list of dictionaries for executed kernel configurations and their execution times.
        :rtype: dict()
        """
        logging.debug("parallel compile runner started for " + self.kernel_options.kernel_name)

        elements = list(parameter_space)
        results = []
        worker_time = 0
        warmup_time = 0
        for start in range(0, len(elements), self.chunk_size):
            chunk = elements[start : start + self.chunk_size]

            build_timer = Timer()
            prebuilt = self._build(chunk, tuning_options)
            worker_time += build_timer.get()

            chunk_results, chunk_worker_time, chunk_warmup_time = self._evaluate(chunk, tuning_options, prebuilt)
            results += chunk_results
            worker_time += chunk_worker_time
            warmup_time += chunk_warmup_time

        self._add_strategy_and_framework_time(results, worker_time, warmup_time)
        return results

    def _build(self, chunk, tuning_options):
        """Build the kernels of the configurations in chunk that will be evaluated, in parallel.

        :returns: The kernel instance and build of each configuration, indexed by its position in the chunk.
        :rtype: dict(int: tuple(KernelInstance, KernelBuild))
        """
        # Only build configurations that will be evaluated within the budget of function evaluations,
        # and that are not in the cache. A time limit may still be reached while benchmarking.
        budget = tuning_options.budget
        if budget.is_done():
            return {}
        remaining = budget.get_evaluations_remaining()

        instances = {}
        futures = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            for index, element in enumerate(chunk):
                if index >= remaining:
                    break
                x_int = ",".join([str(i) for i in element])
                if tuning_options.cache and x_int in tuning_options.cache:
                    continue

                # kernel instances are created in the main thread, as this may involve writing files
                params = dict(zip(tuning_options.tune_params.keys(), element))
                instance = self.dev.create_kernel_instance(
                    self.kernel_source, self.kernel_options, params, tuning_options.verbose
                )
                instances[index] = instance
                if isinstance(instance, ErrorConfig):
                    continue
                gpu_args = self.dev.preprocess_gpu_arguments(self.gpu_args, params)
                futures[index] = pool.submit(self.dev.build_kernel, instance, gpu_args)

        # all builds have finished when the pool is shut down at the end of the with statement
        return {index: (instance, futures[index].result() if index in futures else None)
                for index, instance in instances.items()}
