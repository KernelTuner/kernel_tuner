import ray
import sys
from ray.util.actor_pool import ActorPool
from time import perf_counter
from collections import deque
import copy

from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.runner import Runner
from kernel_tuner.util import get_num_devices, GPUTypeMismatchError
from kernel_tuner.runners.ray.cache_manager import CacheManager
from kernel_tuner.strategies.common import create_actor_on_device, initialize_ray


class ParallelRunner(Runner):
    """ParallelRunner is used for tuning with multiple processes/threads using Ray for distributed computing."""

    def __init__(
        self,
        kernel_source,
        kernel_options,
        device_options,
        iterations,
        observers,
        num_gpus=None,
        cache_manager=None,
        actors=None,
        simulation_mode=False,
    ):
        """Instantiate the ParallelRunner.

        :param kernel_source: The kernel source
        :type kernel_source: kernel_tuner.core.KernelSource

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param device_options: A dictionary with all options for the device
            on which the kernel should be tuned.
        :type device_options: kernel_tuner.interface.Options

        :param iterations: The number of iterations used for benchmarking
            each kernel instance.
        :type iterations: int

        :param observers: List of observers.
        :type observers: list

        :param num_gpus: Number of GPUs to use. Defaults to None.
        :type num_gpus: int, optional

        :param cache_manager: Cache manager instance. Defaults to None.
        :type cache_manager: kernel_tuner.runners.ray.cache_manager.CacheManager, optional

        :param actors: List of pre-initialized actors. Defaults to None.
        :type actors: list, optional

        :param simulation_mode: Flag to indicate simulation mode. Defaults to False.
        :type simulation_mode: bool, optional
        """
        self.dev = (
            DeviceInterface(kernel_source, iterations=iterations, observers=observers, **device_options)
            if not simulation_mode
            else None
        )
        self.kernel_source = kernel_source
        self.simulation_mode = simulation_mode
        self.kernel_options = kernel_options
        self.start_time = perf_counter()
        self.last_strategy_start_time = self.start_time
        self.observers = observers
        self.iterations = iterations
        self.device_options = device_options
        self.cache_manager = cache_manager
        self.num_gpus = num_gpus
        self.actors = actors

        initialize_ray()

        if num_gpus is None:
            self.num_gpus = get_num_devices(simulation_mode)

        # So we know the number of GPUs in the cache file
        if not simulation_mode:
            self.dev.name = [self.dev.name] * self.num_gpus

    def get_environment(self, tuning_options):
        return self.dev.get_environment()

    def run(self, parameter_space=None, tuning_options=None, ensemble=None, searchspace=None, cache_manager=None):
        """Run the tuning process with parallel execution.

        :param parameter_space: The parameter space to explore.
        :type parameter_space: iterable

        :param tuning_options: Tuning options. Defaults to None.
        :type tuning_options: dict, optional

        :param ensemble: List of strategies for ensemble. Defaults to None.
        :type ensemble: list, optional

        :param searchspace: The search space to explore. Defaults to None.
        :type searchspace: kernel_tuner.searchspace.Searchspace, optional

        :param cache_manager: Cache manager instance. Defaults to None.
        :type cache_manager: kernel_tuner.runners.ray.cache_manager.CacheManager, optional

        :returns: Results of the tuning process.
        :rtype: list of dict
        """
        if (
            tuning_options is None
        ):  # HACK as tuning_options can't be the first argument and parameter_space needs to be a default argument
            raise ValueError("tuning_options cannot be None")

        # Create RemoteActor instances
        if self.actors is None:
            runner_attributes = [
                self.kernel_source,
                self.kernel_options,
                self.device_options,
                self.iterations,
                self.observers,
            ]
            self.actors = [
                create_actor_on_device(
                    *runner_attributes, id=_id, cache_manager=self.cache_manager, simulation_mode=self.simulation_mode
                )
                for _id in range(self.num_gpus)
            ]

        # Check if all GPUs are of the same type
        if not self.simulation_mode and not self._check_gpus_equals():
            raise GPUTypeMismatchError(f"Different GPU types found")

        if self.cache_manager is None:
            if cache_manager is None:
                cache_manager = CacheManager.remote(tuning_options.cache, tuning_options.cachefile)
            self.cache_manager = cache_manager

        # set the cache manager for each actor. Can't be done in constructor because we do not always yet have the tuning_options
        for actor in self.actors:
            actor.set_cache_manager.remote(self.cache_manager)

        # Some observers can't be pickled
        run_tuning_options = copy.deepcopy(tuning_options)
        run_tuning_options["observers"] = None
        # Determine what type of parallelism and run appropriately
        if parameter_space and not ensemble and not searchspace:
            results, tuning_options_list = self.parallel_function_evaluation(run_tuning_options, parameter_space)
        elif ensemble and searchspace and not parameter_space:
            results, tuning_options_list = self.multi_strategy_parallel_execution(
                ensemble, run_tuning_options, searchspace
            )
        else:
            raise ValueError("Invalid arguments to parallel runner run method")

        # Update tuning options
        # NOTE: tuning options won't have the state of the observers created in the actors as they can't be pickled
        cache, cachefile = ray.get(self.cache_manager.get_cache.remote())
        tuning_options.cache = cache
        tuning_options.cachefile = cachefile
        if self.simulation_mode:
            tuning_options.simulated_time += self._calculate_simulated_time(tuning_options_list)

        return results

    def multi_strategy_parallel_execution(self, ensemble, tuning_options, searchspace):
        """Runs strategies from the ensemble in parallel using distributed actors,
        manages dynamic task allocation, and collects results.

        :param ensemble: List of strategies to execute.
        :type ensemble: list

        :param tuning_options: Tuning options.
        :type tuning_options: dict

        :param searchspace: Search space to explore.
        :type searchspace: kernel_tuner.searchspace.Searchspace

        :returns: Processed results and tuning options list.
        :rtype: tuple
        """
        ensemble_queue = deque(ensemble)
        pending_tasks = {}
        all_results = []
        options = tuning_options.strategy_options
        max_feval = options["max_fevals"]
        num_strategies = len(ensemble)

        # distributing feval to all strategies
        base_eval_per_strategy = max_feval // num_strategies
        remainder = max_feval % num_strategies
        evaluations_per_strategy = [base_eval_per_strategy] * num_strategies
        for i in range(remainder):
            evaluations_per_strategy[i] += 1

        # Ensure we always have a list of search spaces
        searchspaces = [searchspace] * num_strategies
        searchspaces = deque(searchspaces)

        # Start initial tasks for each actor
        for actor in self.actors:
            strategy = ensemble_queue.popleft()
            searchspace = searchspaces.popleft()
            remote_tuning_options = self._setup_tuning_options(tuning_options, evaluations_per_strategy)
            task = actor.execute.remote(
                strategy=strategy, searchspace=searchspace, tuning_options=remote_tuning_options
            )
            pending_tasks[task] = actor

        # Manage task completion and redistribution
        while pending_tasks:
            done_ids, _ = ray.wait(list(pending_tasks.keys()), num_returns=1)
            for done_id in done_ids:
                result = ray.get(done_id)
                all_results.append(result)
                actor = pending_tasks.pop(done_id)

                # Reassign actors if strategies remain
                if ensemble_queue:
                    strategy = ensemble_queue.popleft()
                    searchspace = searchspaces.popleft()
                    remote_tuning_options = self._setup_tuning_options(tuning_options, evaluations_per_strategy)
                    task = actor.execute.remote(
                        strategy=strategy, searchspace=searchspace, tuning_options=remote_tuning_options
                    )
                    pending_tasks[task] = actor

        # Process results
        results, tuning_options_list = self._process_results_ensemble(all_results)

        return results, tuning_options_list

    def _setup_tuning_options(self, tuning_options, evaluations_per_strategy):
        """Set up tuning options for each strategy in the ensemble.

        :param tuning_options: Original tuning options.
        :type tuning_options: dict

        :param evaluations_per_strategy: Number of evaluations per strategy.
        :type evaluations_per_strategy: list

        :returns: Modified tuning options.
        :rtype: dict
        """
        new_tuning_options = copy.deepcopy(tuning_options)
        new_tuning_options.strategy_options["max_fevals"] = evaluations_per_strategy.pop(0)
        # the stop criterion uses the max feval in tuning options for some reason
        new_tuning_options["max_fevals"] = new_tuning_options.strategy_options["max_fevals"]
        return new_tuning_options

    def _process_results_ensemble(self, all_results):
        """Process the results from the ensemble execution.

        :param all_results: List of results from all strategies.
        :type all_results: list

        :returns: Processed results and tuning options list.
        :rtype: tuple
        """
        results = []
        tuning_options_list = []

        for strategy_results, tuning_options in all_results:
            results.extend(strategy_results)
            tuning_options_list.append(tuning_options)

        return results, tuning_options_list

    def parallel_function_evaluation(self, tuning_options, parameter_space):
        """Perform parallel function evaluation.

        :param tuning_options: Tuning options.
        :type tuning_options: dict

        :param parameter_space: Parameter space to explore.
        :type parameter_space: list

        :returns: Results and tuning options list.
        :rtype: tuple
        """
        # Create a pool of RemoteActor actors
        self.actor_pool = ActorPool(self.actors)
        # Distribute execution of the `execute` method across the actor pool with varying parameters and tuning options, collecting the results asynchronously.
        all_results = list(
            self.actor_pool.map_unordered(lambda a, v: a.execute.remote(tuning_options, element=v), parameter_space)
        )
        results = [x[0] for x in all_results]
        tuning_options_list = [x[1] for x in all_results]
        return results, tuning_options_list

    def _process_results(self, all_results, searchspace):
        """
        Process the results and remove duplicates based on the searchspace.
        """
        unique_configs = set()
        final_results = []

        for strategy_results, tuning_options in all_results:
            for new_result in strategy_results:
                config_signature = tuple(new_result[key] for key in searchspace.tune_params)
                if config_signature not in unique_configs:
                    final_results.append(new_result)
                    unique_configs.add(config_signature)
        return final_results

    def _calculate_simulated_time(self, tuning_options_list):
        """
        Calculate the maximum simulated time from the list of tuning options.
        """
        simulated_times = []
        for tuning_options in tuning_options_list:
            simulated_times.append(tuning_options.simulated_time)
        return max(simulated_times)

    def _check_gpus_equals(self):
        """
        Check if all GPUs are of the same type.
        """
        gpu_types = []
        env_refs = [actor.get_environment.remote() for actor in self.actors]
        environments = ray.get(env_refs)
        for env in environments:
            gpu_types.append(env["device_name"])
        if len(set(gpu_types)) == 1:
            print(f"Running on {len(gpu_types)} {gpu_types[0]}", file=sys.stderr)
            return True
        else:
            return False

    def clean_up_ray(self):
        """
        Clean up Ray actors and cache manager.
        """
        if self.actors is not None:
            for actor in self.actors:
                ray.kill(actor)
        if self.cache_manager is not None:
            ray.kill(self.cache_manager)
