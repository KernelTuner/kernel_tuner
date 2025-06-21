"""Strategy that dynamically imports and enables the use of pyATF strategies."""

from importlib import import_module
import zlib
from pathlib import Path

from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies import common
from kernel_tuner.strategies.common import CostFunc
from kernel_tuner.util import StopCriterionReached

supported_searchtechniques = ["auc_bandit", "differential_evolution", "pattern_search", "round_robin", "simulated_annealing", "torczon"]

_options = dict(searchtechnique=(f"PyATF optimization algorithm to use, choose any from {supported_searchtechniques}", "simulated_annealing"))

def get_cache_checksum(d: dict):
    checksum=0
    for item in d.items():
        c1 = 1
        for t in item:
            c1 = zlib.adler32(bytes(repr(t),'utf-8'), c1)
        checksum=checksum ^ c1
    return checksum

def tune(searchspace: Searchspace, runner, tuning_options):
    from pyatf.search_techniques.search_technique import SearchTechnique
    from pyatf.search_space import SearchSpace as pyATFSearchSpace
    from pyatf import TP
    try:
        import dill
        pyatf_search_space_caching = True
    except ImportError:
        from warnings import warn
        pyatf_search_space_caching = False
        warn("dill is not installed, pyATF search space caching will not be used.")

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

    # get the search space hash
    tune_params_hashable = {k: ",".join([str(i) for i in v]) if isinstance(v, (list, tuple)) else v for k, v in searchspace.tune_params.items()}
    searchspace_caches_folder = Path("./pyatf_searchspace_caches")
    searchspace_caches_folder.mkdir(parents=True, exist_ok=True)
    searchspace_cache_path = searchspace_caches_folder / Path(f"pyatf_searchspace_cache_{get_cache_checksum(tune_params_hashable)}.pkl")

    # initialize the search space
    if not pyatf_search_space_caching or not searchspace_cache_path.exists():
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
        if pyatf_search_space_caching:
            dill.dump(search_space_pyatf, open(searchspace_cache_path, "wb"))
    elif searchspace_cache_path.exists():
        search_space_pyatf = dill.load(open(searchspace_cache_path, "rb"))

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


tune.__doc__ = common.get_strategy_docstring("pyatf_strategies", _options)
