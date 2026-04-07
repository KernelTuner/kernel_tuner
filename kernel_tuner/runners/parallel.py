"""A specialized runner that tunes in parallel the parameter space."""
from collections import deque
import logging
import socket
from time import perf_counter
from typing import List, Optional
from kernel_tuner.core import DeviceInterface
from kernel_tuner.interface import Options
from kernel_tuner.runners.runner import Runner
from kernel_tuner.util import (
    Timer,
    disable_benchmark_timings,
    ErrorConfig,
    TuningBudget,
    print_config_output,
    process_metrics,
    store_cache,
)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    import ray
except ImportError as e:
    raise ImportError(f"unable to initialize the parallel runner: {e}") from e


@ray.remote(num_gpus=1)
class WorkerActor:
    def __init__(
        self, kernel_source, kernel_options, device_options, tuning_options, iterations, observers
    ):
        # detect language and create high-level device interface
        self.dev = DeviceInterface(
            kernel_source, iterations=iterations, observers=observers, **device_options
        )

        self.units = self.dev.units
        self.quiet = device_options.quiet
        self.kernel_source = kernel_source
        self.warmed_up = False if self.dev.requires_warmup else True
        self.kernel_options = kernel_options
        self.tuning_options = tuning_options

        # move data to the GPU
        self.gpu_args = self.dev.ready_argument_list(kernel_options.arguments)

    def shutdown(self):
        ray.actor.exit_actor()

    def get_environment(self):
        # Get the device properties
        env = dict(self.dev.get_environment())

        # Get the host name
        env["host_name"] = socket.gethostname()

        # Get info about the ray instance
        ctx = ray.get_runtime_context()
        env["ray"] = {
            "node_id": ctx.get_node_id(),
            "worker_id": ctx.get_worker_id(),
            "actor_id": ctx.get_actor_id(),
        }

        return env

    def run(self, params):
        # TODO: logging.debug("sequential runner started for " + self.kernel_options.kernel_name)
        result = None

        # attempt to warmup the GPU by running the first config in the parameter space and ignoring the result
        if not self.warmed_up:
            self.dev.compile_and_benchmark(
                self.kernel_source, self.gpu_args, params, self.kernel_options, self.tuning_options
            )
            self.warmed_up = True

        result = self.dev.compile_and_benchmark(
            self.kernel_source, self.gpu_args, params, self.kernel_options, self.tuning_options
        )

        params.update(result)
        params["timestamp"] = datetime.now(timezone.utc).isoformat()
        params["ray_actor_id"] = ray.get_runtime_context().get_actor_id()
        params["host_name"] = socket.gethostname()

        # all visited configurations are added to results to provide a trace for optimization strategies
        return params


class Worker:
    def __init__(self, index, actor):
        self.index = index
        self.running_jobs = []
        self.maximum_running_jobs = 2
        self.is_running = True
        self.actor = actor
        self.env = ray.get(actor.get_environment.remote())

    def __repr__(self):
        try:
            actor_id = self.env["ray"]["actor_id"]
            host_name = self.env["host_name"]
            return f"{self.index} ({host_name}, {actor_id})"
        except Exception:
            return f"{self.index}"

    def shutdown(self):
        if not self.is_running:
            return

        self.is_running = False

        try:
            self.actor.shutdown.remote()
        except Exception:
            logger.exception("failed to request actor shutdown: worker %s", self)

    def submit(self, config):
        job = self.actor.run.remote(config)
        self.running_jobs.append(job)
        return job

    def is_available(self):
        if not self.is_running:
            return False

        # Check for ready jobs, but do not block
        _, self.running_jobs = ray.wait(self.running_jobs, timeout=0)

        # Available if this actor can now run another job
        return len(self.running_jobs) < self.maximum_running_jobs


def launch_workers(n, *args):
    actors = []
    workers = []

    try:
        # Start all actors in parallel
        for _ in range(n):
            actors.append(WorkerActor.remote(*args))

        # Create `Worker` objects. This blocks until each worker is ready
        for index, actor in enumerate(actors):
            worker = Worker(index, actor)
            workers.append(worker)
            logging.info("connected: worker %s", worker)

        return workers
    except:
        # Attempt to shut down actors
        for actor in actors:
            try:
                actor.shutdown.remote()
            except:
                logger.exception("failed to request actor shutdown: %s", actor)
        raise


class ParallelRunner(Runner):
    def __init__(
        self,
        kernel_source,
        kernel_options,
        device_options,
        tuning_options,
        iterations,
        observers,
        num_workers=None,
    ):
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        if num_workers is None:
            num_workers = int(ray.cluster_resources().get("GPU", 0))

        if num_workers == 0:
            raise RuntimeError("failed to initialize parallel runner: no GPUs found")

        if num_workers < 1:
            raise RuntimeError(
                f"failed to initialize parallel runner: invalid number of GPUs specified: {num_workers}"
            )

        self.workers = launch_workers(
            num_workers,
            kernel_source,
            kernel_options,
            device_options,
            tuning_options,
            iterations,
            observers,
        )

        # Check if all workers have the same device
        device_names = {str(w.env.get("device_name")) for w in self.workers}
        if len(device_names) != 1:
            raise RuntimeError(
                f"failed to initialize parallel runner: workers have different devices: {sorted(device_names)}"
            )

        self.device_name = device_names.pop()

        # TODO: Get units from the device?
        self.units = {"time": "ms"}
        self.quiet = device_options.quiet

        # Print some debugging information
        if tuning_options.verbose:
            print(f"parallel tuning on {self.device_name} with {num_workers} workers")
            for worker in self.workers:
                print(f" - worker {worker}")

    def get_device_info(self):
        # TODO: Get max_threads from the device?
        return Options({"name": self.device_name, "max_threads": 1024})

    def get_environment(self, tuning_options):
        return {"device_name": self.device_name, "workers": [w.env for w in self.workers]}

    def shutdown(self):
        for worker in self.workers:
            try:
                worker.shutdown()
            except Exception as err:
                logger.exception(f"error while shutting down worker {worker}")

    def available_parallelism(self):
        return len(self.workers)

    def find_available_worker(self) -> Optional[Worker]:
        for i, worker in enumerate(list(self.workers)):
            if worker.is_available():
                # Push worker to back of list
                self.workers.pop(i)
                self.workers.append(worker)
                return worker

        return None

    def submit_jobs(self, jobs, budget: TuningBudget):
        pending_jobs = deque(jobs)
        running_jobs = dict()
        last_submit_timer = Timer()

        while pending_jobs or running_jobs:
            # If there are pending jobs, try to submit one
            if pending_jobs:
                # If there are still pending jobs and the budget has been exceeded.
                # We return `None` to indicate that no result is available for these jobs.
                if budget.is_done():
                    key, config = pending_jobs.popleft()
                    yield (key, None)
                    continue

                # Find a worker that is available
                worker = self.find_available_worker()
                if worker is not None:
                    # Pop job and submit it
                    key, config = pending_jobs.popleft()
                    ref = worker.submit(config)
                    last_submit_timer = Timer()
                    running_jobs[ref] = (key, worker, last_submit_timer)

                    logger.info(f"job submitted to worker {worker}: {key}")
                    budget.add_evaluations(1)
                    continue

            # If we there pending jobs left but no running jobs and
            # no available worker, then we are in an invalid state.
            if not running_jobs:
                raise RuntimeError("invalid state: no ray workers available")

            # Wait for jobs to finish
            ready_jobs, _ = ray.wait(list(running_jobs), num_returns=1, timeout=60)

            # Process finished jobs
            for job in ready_jobs:
                key, worker, timer = running_jobs.pop(job)
                logger.info(f"job finished on worker {worker}: {key} (took {timer})")
                yield (key, ray.get(job))

            # No ready jobs? Timeout expired. Print warning
            if not ready_jobs:
                print(
                    f"warning: no progress made on {len(running_jobs)} jobs in {last_submit_timer}, are we stuck?"
                )

                for key, worker, timer in running_jobs.values():
                    print(f"- job {key} submitted to worker {worker}: {timer.get():.2f} ago")

    def run(self, parameter_space, tuning_options) -> List[Optional[dict]]:
        metrics = tuning_options.metrics
        objective = tuning_options.objective

        jobs = []  # Jobs that need to be executed
        results = []  # Results that will be returned at the end
        key2index = dict()  # Used to insert job result back into `results`
        duplicate_entries = []  # Stores (i, j) if `i` is a duplicate of `j`.

        # Select jobs which are not in the cache
        for index, config in enumerate(parameter_space):
            params = dict(zip(tuning_options.tune_params.keys(), config))
            key = ",".join([str(i) for i in config])

            # Element is in cache
            if key in tuning_options.cache:
                # We must disable the timings as otherwise these will counted
                # as part of the total_compile/benchmark/verification_time
                result = disable_benchmark_timings(tuning_options.cache[key])

                # recompute matrics for this entry
                result = process_metrics(result, metrics)

                results.append(result)

            # Element is duplicate entry in `parameter_space`
            elif key in key2index:
                duplicate_entries.append((index, key2index[key]))
                results.append(None)

            # Element must become a job
            else:
                key2index[key] = index
                jobs.append((key, params))
                results.append(None)

        total_worker_time = 0

        # Submit jobs and wait for them to finish
        for key, result in self.submit_jobs(jobs, tuning_options.budget):
            # `None` indicate that no result is available since the budget is exceeded.
            # We can skip it, meaning that `results` contains `None`s for these entries
            if result is None:
                continue

            # Collect total time spent by worker
            total_worker_time += (
                result["compile_time"] + result["verification_time"] + result["benchmark_time"]
            ) / 1000

            # only compute metrics on configs that have not errored
            if not isinstance(result.get(objective), ErrorConfig):
                result = process_metrics(result, metrics)
            else:
                logging.error(
                    "kernel configuration {key} was skipped silently due to compile or runtime failure",
                    key,
                )

            # print configuration to the console
            print_config_output(
                tuning_options.tune_params, result, self.quiet, tuning_options.metrics, self.units
            )

            # add configuration to cache
            store_cache(key, result, tuning_options.cachefile, tuning_options.cache)

            # Store the result into the output array
            results[key2index[key]] = result

        # Fix duplicate entries. Duplicate entires do not get benchmark timings
        # as otherwise we would count them multiple times in the total
        for i, j in duplicate_entries:
            if results[j]:
                results[i] = disable_benchmark_timings(results[j])

        # Count the number of valid results
        num_valid_results = sum(bool(r) for r in results)

        # If there are valid results, set timings
        if num_valid_results > 0:
            total_time = self.timer.get_and_reset()

            strategy_time = self.accumulated_strategy_time
            self.accumulated_strategy_time = 0

            runner_time = total_time - strategy_time
            framework_time = max(runner_time * len(self.workers) - total_worker_time, 0)

            # Amortize the time over all the results
            for result in results:
                if result:
                    # Time must be in ms
                    result["strategy_time"] = strategy_time / num_valid_results
                    result["framework_time"] = framework_time / num_valid_results

        return results
