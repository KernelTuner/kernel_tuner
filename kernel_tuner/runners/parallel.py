"""A specialized runner that tunes in parallel the parameter space."""
import logging
import socket
from time import perf_counter
from kernel_tuner.interface import Options
from kernel_tuner.util import ErrorConfig, print_config, print_config_output, process_metrics, store_cache
from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.runner import Runner
from datetime import datetime, timezone

try:
    import ray
except ImportError as e:
    raise Exception(f"Unable to initialize the parallel runner: {e}")


@ray.remote(num_gpus=1)
class DeviceActor:
    def __init__(self, kernel_source, kernel_options, device_options, tuning_options, iterations, observers):
        # detect language and create high-level device interface
        self.dev = DeviceInterface(kernel_source, iterations=iterations, observers=observers, **device_options)

        self.units = self.dev.units
        self.quiet = device_options.quiet
        self.kernel_source = kernel_source
        self.warmed_up = False if self.dev.requires_warmup else True
        self.start_time = perf_counter()
        self.last_strategy_start_time = self.start_time
        self.last_strategy_time = 0
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
        ray_info = ray.get_runtime_context()
        env["ray"] = dict(
            node_id=ray_info.get_node_id(),
            worker_id=ray_info.get_worker_id(),
            actor_id=ray_info.get_actor_id(),
        )

        return env

    def run(self, element):
        # TODO: logging.debug("sequential runner started for " + self.kernel_options.kernel_name)
        objective = self.tuning_options.objective
        metrics = self.tuning_options.metrics

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

        if isinstance(result.get(objective), ErrorConfig):
            logging.debug("kernel configuration was skipped silently due to compile or runtime failure")

        params.update(result)

        # only compute metrics on configs that have not errored
        if metrics and not isinstance(params.get(objective), ErrorConfig):
            params = process_metrics(params, metrics)

        # get the framework time by estimating based on other times
        total_time = 1000 * ((perf_counter() - self.start_time) - warmup_time)
        params["strategy_time"] = self.last_strategy_time
        params["framework_time"] = max(
            total_time
            - (
                params["compile_time"]
                + params["verification_time"]
                + params["benchmark_time"]
                + params["strategy_time"]
            ),
            0,
        )

        params["timestamp"] = str(datetime.now(timezone.utc))
        params["ray_actor_id"] = ray.get_runtime_context().get_actor_id()
        params["host_name"] = socket.gethostname()

        self.start_time = perf_counter()

        # all visited configurations are added to results to provide a trace for optimization strategies
        return params


class DeviceActorState:
    def __init__(self, actor):
        self.actor = actor
        self.running_jobs = []
        self.maximum_running_jobs = 1
        self.is_running = True
        self.env = ray.get(actor.get_environment.remote())

    def __repr__(self):
        actor_id = self.env["ray"]["actor_id"]
        host_name = self.env["host_name"]
        return f"{actor_id} ({host_name})"

    def shutdown(self):
        if self.is_running:
            self.is_running = False
            self.actor.shutdown.remote()

    def submit(self, *args):
        job = self.actor.run.remote(*args)
        self.running_jobs.append(job)
        return job

    def is_available(self):
        if not self.is_running:
            return False

        # Check for ready jobs, but do not block
        ready_jobs, self.running_jobs = ray.wait(self.running_jobs, timeout=0)
        ray.get(ready_jobs)

        # Available if this actor can now run another job
        return len(self.running_jobs) < self.maximum_running_jobs


class ParallelRunner(Runner):
    def __init__(self, kernel_source, kernel_options, device_options, tuning_options, iterations, observers, num_workers=None):
        if not ray.is_initialized():
            ray.init()

        if num_workers is None:
            num_workers = int(ray.cluster_resources().get("GPU", 0))

        if num_workers == 0:
            raise RuntimeError("failed to initialize parallel runner: no GPUs found")

        if num_workers < 1:
            raise RuntimeError(f"failed to initialize parallel runner: invalid number of GPUs specified: {num_workers}")

        self.workers = []

        try:
            for index in range(num_workers):
                actor = DeviceActor.remote(kernel_source, kernel_options, device_options, tuning_options, iterations, observers)
                worker = DeviceActorState(actor)
                self.workers.append(worker)

                logging.info(f"launched worker {index}: {worker}")
        except:
            # If an exception occurs, shut down the worker
            self.shutdown()
            raise

        # Check if all workers have the same device
        device_names = {w.env.get("device_name") for w in self.workers}
        if len(device_names) != 1:
            self.shutdown()
            raise RuntimeError(
                f"failed to initialize parallel runner: workers have different devices: {sorted(device_names)}"
            )

        self.device_name = device_names.pop()

        # TODO: Get this from the device
        self.units = {"time": "ms"}
        self.quiet = device_options.quiet

    def get_device_info(self):
        return Options(dict(max_threads=1024))

    def get_environment(self, tuning_options):
        return dict(
            device_name=self.device_name,
            workers=[w.env for w in self.workers],
        )

    def shutdown(self):
        for worker in self.workers:
            try:
                worker.shutdown()
            except Exception as err:
                logging.warning(f"error while shutting down worker {worker}: {err}")

    def submit_job(self, *args):
        while True:
            # Find an idle actor
            for i, worker in enumerate(list(self.workers)):
                if worker.is_available():
                    # push the worker to the end
                    self.workers.pop(i)
                    self.workers.append(worker)

                    # Submit the work
                    return worker.submit(*args)

            # Gather all running jobs
            running_jobs = [job for w in self.workers for job in w.running_jobs]

            # If there are no running jobs, then something must be wrong.
            # Maybe a worker has crashed or gotten into an invalid state.
            if not running_jobs:
                raise Exception("invalid state: no Ray workers are available to run job")

            # Wait until any running job completes
            ray.wait(running_jobs, num_returns=1)

    def run(self, parameter_space, tuning_options):
        running_jobs = dict()
        completed_jobs = dict()

        # Submit jobs which are not in the cache
        for config in parameter_space:
            params = dict(zip(tuning_options.tune_params.keys(), config))
            key = ",".join([str(i) for i in config])

            if key in tuning_options.cache:
                completed_jobs[key] = tuning_options.cache[key]
            else:
                assert key not in running_jobs
                running_jobs[key] = self.submit_job(params)
                completed_jobs[key] = None

        # Wait for the running jobs to finish
        for key, job in running_jobs.items():
            result = ray.get(job)
            completed_jobs[key] = result

            # print configuration to the console
            print_config_output(tuning_options.tune_params, result, self.quiet, tuning_options.metrics, self.units)

            # add configuration to cache
            store_cache(key, result, tuning_options.cachefile, tuning_options.cache)

        return list(completed_jobs.values())
