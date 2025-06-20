"""Strategy that dynamically imports and enables the use of pyATF strategies."""

from importlib import import_module

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc
from kernel_tuner.util import StopCriterionReached

supported_searchtechniques = ["auc_bandit", "differential_evolution", "pattern_search", "round_robin", "simulated_annealing", "torczon"]

_options = dict(searchtechnique=(f"PyATF optimization algorithm to use, choose any from {supported_searchtechniques}", "simulated_annealing"))

def tune(searchspace: Searchspace, runner, tuning_options):
    from pyatf.search_techniques.search_technique import SearchTechnique
    from pyatf.search_space import SearchSpace as pyATFSearchSpace

    # setup the Kernel Tuner functionalities
    cost_func = CostFunc(searchspace, tuning_options, runner, scaling=False, snap=False, return_invalid=False)

    # dynamically import the search technique based on the provided options
    module_name,  = common.get_options(tuning_options.strategy_options, _options)
    module = import_module(f"pyatf.search_techniques.{module_name}")
    class_name = [d for d in dir(module) if d.lower() == module_name.replace('_','')][0]
    searchtechnique_class = getattr(module, class_name)

    # instantiate the search technique
    search_technique = searchtechnique_class()
    search_technique.initialize(len(searchspace.param_names))
    assert isinstance(search_technique, SearchTechnique), f"Search technique {search_technique} is not a valid pyATF search technique."

    # initialize the search space
    searchspace_pyatf = Searchspace(
        searchspace.tune_params, 
        tuning_options.restrictions_unmodified, 
        searchspace.max_threads, 
        searchspace.block_size_names, 
        defer_construction=True,
        framework="pyatf"
    )
    tune_params_pyatf = searchspace_pyatf.get_tune_params_pyatf()
    assert isinstance(tune_params_pyatf, (tuple, list)), f"Tuning parameters must be a tuple or list of tuples, is {type(tune_params_pyatf)} ({tune_params_pyatf})."
    search_space_pyatf = pyATFSearchSpace(*tune_params_pyatf, enable_1d_access=False) # SearchTechnique1D currently not supported

    # initialize
    get_next_coordinates_or_indices = search_technique.get_next_coordinates
    coordinates_or_indices = set()  # Set[Union[Coordinates, Index]]
    costs = {}   # Dict[Union[Coordinates, Index], Cost]
    eval_count = 0

    try:
        # optimization loop (KT-compatible re-implementation of `make_step` from TuningRun)
        while eval_count < searchspace.size:

            # get new coordinates
            if not coordinates_or_indices:
                if costs:
                    search_technique.report_costs(costs)
                    costs.clear()
                coordinates_or_indices.update(get_next_coordinates_or_indices())

            # get configuration
            coords_or_index = coordinates_or_indices.pop()
            config = search_space_pyatf.get_configuration(coords_or_index)
            valid = True
            cost = None

            # evaluate the configuration
            x = tuple([config[k] for k in searchspace.tune_params.keys()])
            opt_result = cost_func(x, check_restrictions=False)

            # adjust opt_result to expected PyATF output in cost and valid
            if not isinstance(opt_result, (int, float)):
                valid = False
            else:
                cost = opt_result
                eval_count += 1

            # record the evaluation
            costs[coords_or_index] = cost
    except StopCriterionReached:
        pass
    finally:
        search_technique.finalize()

    return cost_func.results


# class TuningRun:
#     def __init__(self,
#                     search_space: SearchSpace | Tuple[TP, ...],
#                     cost_function: CostFunction,
#                     search_technique: Optional[Union[SearchTechnique, SearchTechnique1D]],
#                     verbosity: Optional[int],
#                     log_file: Optional[str],
#                     abort_condition: Optional[AbortCondition]):
#         if search_space is None:
#             raise ValueError('missing call to `Tuner.tuning_parameters(...)`: no tuning parameters defined')

#         # tuning data
#         self._search_space: SearchSpace
#         self._search_technique: SearchTechnique | SearchTechnique1D
#         self._abort_condition: AbortCondition  # TODO: does not work (add initialization)
#         self._tps: Tuple[TP, ...]
#         self._tuning_data: Optional[TuningData] = None
#         self._cost_function: CostFunction = cost_function

#         # progress data
#         self._verbosity = verbosity
#         self._log_file: Optional[TextIO] = None
#         self._last_log_dump: Optional[int] = None
#         self._last_line_length: Optional[int] = None
#         self._tuning_start_ns: Optional[int] = None

#         # prepare search technique
#         self._search_technique: SearchTechnique | SearchTechnique1D = search_technique
#         if self._search_technique is None:
#             self._search_technique = AUCBandit()
#         if isinstance(self._search_technique, SearchTechnique):
#             self._get_next_coordinates_or_indices = self._search_technique.get_next_coordinates
#             self._coordinates_or_index_param_name = 'search_space_coordinates'
#         else:
#             self._get_next_coordinates_or_indices = self._search_technique.get_next_indices
#             self._coordinates_or_index_param_name = 'search_space_index'
#         self._coordinates_or_indices: Set[Union[Coordinates, Index]] = set()
#         self._costs: Dict[Union[Coordinates, Index], Cost] = {}

#         # generate search space
#         if isinstance(search_space, SearchSpace):
#             self._search_space = search_space
#         else:
#             self._search_space = SearchSpace(*search_space,
#                                                 enable_1d_access=isinstance(self._search_technique, SearchTechnique1D),
#                                                 verbosity=verbosity)
#         self._tps = self._search_space.tps
#         self._search_space_generation_ns = self._search_space.generation_ns
#         if self._verbosity >= 2:
#             print(f'search space size: {self._search_space.constrained_size}')

#         # prepare abort condition
#         self._abort_condition = abort_condition
#         if self._abort_condition is None:
#             self._abort_condition = Evaluations(len(self._search_space))

#         # open log file
#         if log_file:
#             Path(log_file).parent.mkdir(parents=True, exist_ok=True)
#             self._log_file = open(log_file, 'w')

#     def __del__(self):
#         if self._log_file:
#             self._log_file.close()

#     @property
#     def cost_function(self):
#         return self._cost_function

#     @property
#     def abort_condition(self):
#         return self._abort_condition

#     @property
#     def tuning_data(self):
#         return self._tuning_data

#     def flush_log(self):
#         if self._log_file:
#             self._log_file.seek(0)
#             json.dump(self._tuning_data.to_json(), self._log_file, indent=4)
#             self._log_file.truncate()
#             self._last_log_dump = time.perf_counter_ns()

#     def _print_progress(self, timestamp: datetime, cost: Optional[Cost] = None):
#         now = time.perf_counter_ns()
#         elapsed_ns = now - self._tuning_start_ns
#         elapsed_seconds = elapsed_ns // 1000000000
#         elapsed_time_str = (f'{elapsed_seconds // 3600}'
#                             f':{elapsed_seconds // 60 % 60:02d}'
#                             f':{elapsed_seconds % 60:02d}')
#         progress = self._abort_condition.progress(self._tuning_data)
#         if self._verbosity >= 3:
#             line = (f'\r{timestamp.strftime("%Y-%m-%dT%H:%M:%S")}'
#                     f'    evaluations: {self._tuning_data.number_of_evaluated_configurations}'
#                     f' (valid: {self._tuning_data.number_of_evaluated_valid_configurations})'
#                     f', min. cost: {self._tuning_data.min_cost()}'
#                     f', valid: {cost is not None}'
#                     f', cost: {cost}')
#             line_length = len(line)
#             if line_length < self._last_line_length:
#                 line += ' ' * (self._last_line_length - line_length)
#             print(line)
#         if progress is None:
#             spinner_char = ('-', '\\', '|', '/')[(elapsed_ns // 500000000) % 4]
#             line = f'\rTuning: {spinner_char} {elapsed_time_str}\r'
#             print(line, end='')
#         else:
#             if now > self._tuning_start_ns and progress > 0:
#                 eta_seconds = ceil(((now - self._tuning_start_ns) / progress
#                                     * (1 - progress)) / 1000000000)
#                 eta_str = (f'{eta_seconds // 3600}'
#                             f':{eta_seconds // 60 % 60:02d}'
#                             f':{eta_seconds % 60:02d}')
#             else:
#                 eta_str = '?'
#             filled = '█' * floor(progress * 80)
#             empty = ' ' * ceil((1 - progress) * 80)
#             line = (f'\rexploring search space: |{filled}{empty}|'
#                     f' {progress * 100:6.2f}% {elapsed_time_str} (ETA: {eta_str})')
#             print(line, end='')
#         self._last_line_length = len(line)

#     def initialize(self):
#         # reset progress data
#         self._tuning_start_ns = time.perf_counter_ns()
#         self._last_line_length = 0

#         # create tuning data
#         self._tuning_data = TuningData(list(tp.to_json() for tp in self._tps),
#                                         self._search_space.constrained_size,
#                                         self._search_space.unconstrained_size,
#                                         self._search_space_generation_ns,
#                                         self._search_technique.to_json(),
#                                         self._abort_condition.to_json())

#         # write tuning data
#         self.flush_log()

#         # initialize search technique
#         if isinstance(self._search_technique, SearchTechnique1D):
#             self._search_technique.initialize(len(self._search_space))
#         else:
#             self._search_technique.initialize(self._search_space.num_tps)

#     def make_step(self):
#         # get new coordinates
#         if not self._coordinates_or_indices:
#             if self._costs:
#                 self._search_technique.report_costs(self._costs)
#                 self._costs.clear()
#             self._coordinates_or_indices.update(self._get_next_coordinates_or_indices())

#         # get configuration
#         coords_or_index = self._coordinates_or_indices.pop()
#         config = self._search_space.get_configuration(coords_or_index)

#         # run cost function
#         valid = True
#         try:
#             cost = self._cost_function(config)
#         except CostFunctionError as e:
#             if self._verbosity >= 3:
#                 print('\r' + ' ' * self._last_line_length + '\r', end='')
#                 print('Error raised: ' + e.message)
#                 self._last_line_length = 0
#             cost = None
#             valid = False
#         except BaseException as e:
#             self._tuning_data.record_evaluation(config, False, None, **{
#                 self._coordinates_or_index_param_name: coords_or_index
#             })
#             self.flush_log()
#             raise e
#         timestamp = self._tuning_data.record_evaluation(config, valid, cost, **{
#             self._coordinates_or_index_param_name: coords_or_index
#         })
#         self._costs[coords_or_index] = cost

#         # print progress and dump log file (at most once every 5 minutes)
#         if self._verbosity >= 1:
#             self._print_progress(timestamp, cost)
#         if self._log_file and (self._last_log_dump is None or time.perf_counter_ns() - self._last_log_dump > 3e11):
#             self.flush_log()

#     def finalize(self, sigint_received: bool = False):
#         self._search_technique.finalize()
#         self._tuning_data.record_tuning_finished(sigint_received)

#         # write tuning data to file
#         if self._log_file:
#             self.flush_log()
#             self._log_file.close()
#             self._log_file = None

#         if self._verbosity >= 1:
#             print('\nfinished tuning')
#             if self._verbosity >= 2:
#                 if self._tuning_data.min_cost() is not None:
#                     print('best configuration:')
#                     for tp_name, tp_value in self._tuning_data.configuration_of_min_cost().items():
#                         print(f'    {tp_name} = {tp_value}')
#                     print(f'min cost: {self._tuning_data.min_cost()}')


tune.__doc__ = common.get_strategy_docstring("pyatf_strategies", _options)
