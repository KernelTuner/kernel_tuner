"""A specialized runner that tunes in parallel the parameter space."""
from collections import deque
import logging
import socket
from time import perf_counter
from kernel_tuner.core import DeviceInterface
from kernel_tuner.interface import Options
from kernel_tuner.runners.runner import Runner
from kernel_tuner.util import ErrorConfig, print_config_output, process_metrics, store_cache
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    import ray
except ImportError as e:
    raise ImportError(f"unable to initialize the parallel runner: {e}") from e


@ray.remote(num_gpus=1)
class DeviceActor:
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

    def run(self, key, element):
        # TODO: logging.debug("sequential runner started for " + self.kernel_options.kernel_name)
        params = dict(element)
        result = None
        warmup_time = 0

        # attempt to warmup the GPU by running the first config in the parameter space and ignoring the result
        if not self.warmed_up:
            warmup_time = perf_counter()
            self.dev.compile_and_benchmark(
                self.kernel_source, self.gpu_args, params, self.kernel_options, self.tuning_options
            )
            self.warmed_up = True
            warmup_time = 1e3 * (perf_counter() - warmup_time)

        result = self.dev.compile_and_benchmark(
            self.kernel_source, self.gpu_args, params, self.kernel_options, self.tuning_options
        )

        params.update(result)

        params["timestamp"] = datetime.now(timezone.utc).isoformat()
        params["ray_actor_id"] = ray.get_runtime_context().get_actor_id()
        params["host_name"] = socket.gethostname()

        # all visited configurations are added to results to provide a trace for optimization strategies
        return key, params


class DeviceActorState:
    def __init__(self, index, actor):
        self.index = index
        self.actor = actor
        self.running_jobs = []
        self.maximum_running_jobs = 1
        self.is_running = True
        self.env = ray.get(actor.get_environment.remote())

    def __repr__(self):
        actor_id = self.env["ray"]["actor_id"]
        host_name = self.env["host_name"]
        return f"{self.index} ({host_name}, {actor_id})"

    def shutdown(self):
        if not self.is_running:
            return

        self.is_running = False

        try:
            self.actor.shutdown.remote()
        except Exception:
            logger.exception("Failed to request actor shutdown: %s", self)

    def submit(self, key, config):
        logger.info(f"job submitted to worker {self}: {key}")
        job = self.actor.run.remote(key, config)
        self.running_jobs.append(job)
        return job

    def is_available(self):
        if not self.is_running:
            return False

        # Check for ready jobs, but do not block
        ready_jobs, self.running_jobs = ray.wait(self.running_jobs, timeout=0)

        for job in ready_jobs:
            try:
                key, _result = ray.get(job)
                logger.info(f"job finished on worker {self}: {key}")
            except Exception:
                logger.exception(f"job failed on worker {self}")

        # Available if this actor can now run another job
        return len(self.running_jobs) < self.maximum_running_jobs


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

        self.workers = []

        try:
            # Start workers
            for index in range(num_workers):
                actor = DeviceActor.remote(
                    kernel_source,
                    kernel_options,
                    device_options,
                    tuning_options,
                    iterations,
                    observers,
                )
                worker = DeviceActorState(index, actor)
                self.workers.append(worker)

                logger.info(f"connected to worker {worker}")

            # Check if all workers have the same device
            device_names = {w.env.get("device_name") for w in self.workers}
            if len(device_names) != 1:
                raise RuntimeError(
                    f"failed to initialize parallel runner: workers have different devices: {sorted(device_names)}"
                )
        except:
            # If an exception occurs, shut down the worker and reraise error
            self.shutdown()
            raise

        self.device_name = device_names.pop()

        # TODO: Get units from the device?
        self.start_time = perf_counter()
        self.units = {"time": "ms"}
        self.quiet = device_options.quiet

    def get_device_info(self):
        # TODO: Get this from the device?
        return Options({"max_threads": 1024})

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

    def submit_jobs(self, jobs):
        pending_jobs = deque(jobs)
        running_jobs = []

        while pending_jobs or running_jobs:
            should_wait = True

            # If there is still work left, submit it now
            if pending_jobs:
                for i, worker in enumerate(list(self.workers)):
                    if worker.is_available():
                        # Push worker to back of list
                        self.workers.pop(i)
                        self.workers.append(worker)

                        # Pop job and submit it
                        job = pending_jobs.popleft()
                        ref = worker.submit(*job)
                        running_jobs.append(ref)

                        should_wait = False
                        break

            # If no work was submitted, wait until a worker is available
            if should_wait:
                if not running_jobs:
                    raise RuntimeError("invalid state: no ray workers available")

                ready_jobs, running_jobs = ray.wait(running_jobs, num_returns=1)

                for result in ready_jobs:
                    yield ray.get(result)

    def run(self, parameter_space, tuning_options):
        metrics = tuning_options.metrics
        objective = tuning_options.objective

        jobs = []  # Jobs that need to be executed
        results = []  # Results that will be returned at the end
        key2index = dict()  # Used to insert job result back into `results`
        duplicate_entries = []  # Used for duplicate entries in `parameter_space`

        # Select jobs which are not in the cache
        for index, config in enumerate(parameter_space):
            params = dict(zip(tuning_options.tune_params.keys(), config))
            key = ",".join([str(i) for i in config])

            if key in tuning_options.cache:
                params.update(tuning_options.cache[key])
                params["compile_time"] = 0
                params["verification_time"] = 0
                params["benchmark_time"] = 0
                results.append(params)
            else:
                if key not in key2index:
                    key2index[key] = index
                else:
                    duplicate_entries.append((key2index[key], index))

                jobs.append((key, params))
                results.append(None)

        total_worker_time = 0

        # Submit jobs and wait for them to finish
        for key, result in self.submit_jobs(jobs):
            results[key2index[key]] = result

            # Collect total time spent by worker
            total_worker_time += (
                params["compile_time"] + params["verification_time"] + params["benchmark_time"]
            )

            if isinstance(result.get(objective), ErrorConfig):
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

        # Copy each `i` to `j` for every `i,j` in `duplicate_entries`
        for i, j in duplicate_entries:
            results[j] = dict(results[i])

        total_time = 1000 * (perf_counter() - self.start_time)
        self.start_time = perf_counter()

        strategy_time = self.last_strategy_time
        self.last_strategy_time = 0

        runner_time = total_time - strategy_time
        framework_time = max(runner_time * len(self.workers) - total_worker_time, 0)

        # Post-process all the results
        for params in results:
            # Amortize the time over all the results
            params["strategy_time"] = strategy_time / len(results)
            params["framework_time"] = framework_time / len(results)

            # only compute metrics on configs that have not errored
            if metrics and not isinstance(params.get(objective), ErrorConfig):
                params = process_metrics(params, metrics)

        return results
