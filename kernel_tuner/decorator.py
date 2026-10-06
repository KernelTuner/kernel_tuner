"""Autotune kernels written in Python-embedded DSLs with a decorator, in the spirit of ``triton.autotune``.

The ``autotune`` decorator is placed on top of the decorator of the DSL. The decorated kernel is launched with the
launch syntax of the DSL. When the kernel is launched for a new tuning key, Kernel Tuner tunes the kernel on copies
of the arguments, after which the best configuration is launched. Later launches with the same key directly launch
the best configuration::

    from kernel_tuner import autotune

    @autotune(tune_params={"BLOCK_SIZE": [128, 256, 512, 1024]}, key=["n"])
    @triton.jit
    def vector_add(c_ptr, a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
        ...

    vector_add[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)](c, a, b, n)

How a kernel is launched depends on the DSL:

- Triton: ``kernel[grid](*args)``
- Numba and CuPy: ``kernel[grid, block](*args)``, CuPy also ``kernel(grid, block, args)``
- cuTile: ``kernel[grid](*args)`` or ``kernel.launch(stream, grid, args)``, like ``ct.launch``
- Warp: ``kernel[dim](*args)`` or ``kernel.launch(dim, inputs, ...)``, like ``wp.launch``
- Taichi and CuTe DSL: ``kernel(*args)``
- Tilus: ``Script()(*args)``, constructors with arguments are not supported
- TileLang: ``factory(*factory_args)(*args)``, the factory arguments are part of the tuning key

The grid and thread block dimensions can be functions of a dictionary with the kernel arguments (by name) and the
tunable parameters of the configuration, as in Triton. They can also be given to the decorator, which is needed
when the launch dimensions depend on the tunable parameters in DSLs that do not support functions, such as Numba.

Tuning results can be stored in wisdom files, the format used by Kernel Launcher, see
:mod:`kernel_tuner.utils.wisdom`. A new process then uses the stored results instead of tuning again.

Tuning copies the arguments of the kernel, so the arguments of the user are not modified while tuning. A possible
extension is to support the ``restore_value`` and ``reset_to_zero`` options of ``triton.autotune``, to tune on the
arguments of the user instead of copies.
"""

import atexit
import inspect
import logging
import math
import os
import threading
import warnings

from kernel_tuner.kernel_sources.kernel_source_fn import KernelSourceFn
from kernel_tuner.util import delete_temp_file, get_arg_names, get_kernel_ast
from kernel_tuner.utils import wisdom
from kernel_tuner.utils.language_detection import detect_python_dsl

# attributes in which the DSLs keep the Python function that was decorated, in order of preference
_WRAPPED_FUNCTION_ATTRIBUTES = ("fn", "py_func", "_pyfunc", "_func", "orig_func", "func", "__wrapped__")

# launch options of Triton that can be tunable parameters
_TRITON_LAUNCH_OPTIONS = ("num_warps", "num_stages", "num_ctas", "maxnreg")


def _unwrap(kernel):
    """Return the Python function or class that a DSL decorator wraps.

    Some DSLs return an object that keeps the function in an attribute, others return a wrapper function made with
    functools.wraps, which keeps the function in ``__wrapped__``.
    """
    for _ in range(5):
        if inspect.isclass(kernel):
            return kernel
        for attribute in _WRAPPED_FUNCTION_ATTRIBUTES:
            inner = getattr(kernel, attribute, None)
            if inner is not None and inner is not kernel and (callable(inner) or hasattr(inner, "orig_func")):
                kernel = inner
                break
        else:
            if inspect.isfunction(kernel):
                return kernel
            break
    raise TypeError(f"Could not find the Python function decorated by {type(kernel).__name__}")


def _to_torch(arg):
    """Convert GPU arrays of other libraries to PyTorch tensors that share the memory, which Kernel Tuner copies."""
    import numpy as np
    import torch

    if isinstance(arg, (torch.Tensor, np.ndarray, np.generic)) or not hasattr(arg, "shape"):
        return arg
    if hasattr(arg, "__dlpack__"):
        return torch.from_dlpack(arg)
    if hasattr(arg, "__cuda_array_interface__"):
        return torch.as_tensor(arg, device="cuda")
    return arg


def _expected_like(value, argument):
    """Convert an expected output to the array type of the argument passed to tune_kernel, on the host.

    Kernel Tuner compares the outputs on the host, and requires the expected output to be of the same type.
    """
    import numpy as np
    import torch

    value = _to_torch(value)  # GPU arrays of other libraries
    if isinstance(argument, np.ndarray):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if isinstance(argument, torch.Tensor):
        return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    return value


def _dtype_name(dtype):
    """Return a short name for the data type of an array, Warp uses classes as data types."""
    return dtype.__name__ if isinstance(dtype, type) else str(dtype)


def _convert_tensors(args, convert):
    """Convert the PyTorch tensors in args, used to pass the copies made by Kernel Tuner to the DSL."""
    import torch

    return [convert(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]


class _Launch:
    """How the user launched the kernel: launch dimensions and options of the launch syntax of the DSL."""

    __slots__ = ("grid", "threads", "extra", "options")

    def __init__(self, grid=None, threads=None, extra=(), options=None):
        self.grid = grid
        self.threads = threads
        self.extra = extra  # remaining positional launch options, such as the stream in Numba
        self.options = options or {}  # keyword launch options, such as the stream in cuTile

    def key(self):
        """Launch options that select a different kernel are part of the tuning key."""
        return tuple(self.options.get("factory", ()))


# launch functions per DSL --------------------------------------------------------------------------------------
# These launch the kernel of a configuration. They receive the kernel, the arguments, the tunable parameters that
# are kernel arguments (kwargs), the resolved grid and thread block dimensions, the tunable parameters, and a
# dictionary in which they can keep state per configuration, such as a compiled kernel.


def _launch_triton(kernel, args, kwargs, grid, threads, launch, params, state):
    for option in _TRITON_LAUNCH_OPTIONS:
        if option in params:
            kwargs[option] = params[option]
    return kernel[grid](*args, **kwargs)


def _launch_numba(kernel, args, kwargs, grid, threads, launch, params, state):
    from numba import cuda

    return kernel[(grid, threads, *launch.extra)](*_convert_tensors(args, cuda.as_cuda_array), **kwargs)


def _launch_cupyx(kernel, args, kwargs, grid, threads, launch, params, state):
    import cupy as cp

    args = _convert_tensors(args, cp.from_dlpack)
    return kernel(grid, threads, (*args, *kwargs.values()), **launch.options)


def _launch_warp(kernel, args, kwargs, grid, threads, launch, params, state):
    import warp as wp

    wp.init()
    options = dict(launch.options)
    if "block_dim" in params:
        options["block_dim"] = params["block_dim"]
    elif threads is not None:
        options["block_dim"] = math.prod(threads) if isinstance(threads, (tuple, list)) else threads
    inputs = [*_convert_tensors(args, wp.from_torch), *kwargs.values()]
    return wp.launch(kernel, dim=grid, inputs=inputs, **options)


def _launch_taichi(kernel, args, kwargs, grid, threads, launch, params, state):
    return kernel(*args, **kwargs)


def _launch_cute(kernel, args, kwargs, grid, threads, launch, params, state):
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    args = _convert_tensors(args, from_dlpack)
    if "compiled" not in state:
        state["compiled"] = cute.compile(kernel, *args)
    return state["compiled"](*args, **kwargs)


def _launch_tilus(kernel, args, kwargs, grid, threads, launch, params, state):
    if "script" not in state:
        # Kernel Tuner passes an instance while tuning, the dispatch table holds the class
        state["script"] = kernel() if inspect.isclass(kernel) else kernel
    return state["script"](*args, **kwargs)


def _launch_tilelang(kernel, args, kwargs, grid, threads, launch, params, state):
    if "compiled" not in state:
        factory_args, factory_kwargs = launch.options["factory"]
        state["compiled"] = kernel(*factory_args, **{**dict(factory_kwargs), **kwargs})
    return state["compiled"](*args)


def _launch_cutile(kernel, args, kwargs, grid, threads, launch, params, state):
    import cuda.tile as ct
    import torch

    stream = launch.options.get("stream") or torch.cuda.current_stream()
    with ct.compiler_timeout(60):
        return ct.launch(stream, grid, kernel, (*args, *kwargs.values()))


_LAUNCH_FUNCTIONS = {
    "triton": _launch_triton,
    "numba": _launch_numba,
    "cupyx": _launch_cupyx,
    "warp": _launch_warp,
    "taichi": _launch_taichi,
    "cute": _launch_cute,
    "tilus": _launch_tilus,
    "tilelang": _launch_tilelang,
    "cutile": _launch_cutile,
}

# DSLs that launch kernels with a subscript, kernel[grid](*args), or with Triton-style calling
_SUBSCRIPT_DSLS = {"triton", "numba", "cupyx", "cutile", "warp"}


def _named_arguments(argument_names, args, kwargs, spec):
    """Return the kernel arguments by name, for TileLang these are the arguments of the kernel factory."""
    if "factory" in spec.options:
        factory_args, factory_kwargs = spec.options["factory"]
        return {**dict(zip(argument_names, factory_args)), **dict(factory_kwargs)}
    return {**dict(zip(argument_names, args)), **kwargs}


def _launch_dimensions(grid, threads, spec, argument_names, args, kwargs, params):
    """Return the grid and thread block dimensions, those of the decorator take precedence over those of the launch.

    Dimensions can be functions of a dictionary with the kernel arguments by name and the tunable parameters.
    """
    grid = grid if grid is not None else spec.grid
    threads = threads if threads is not None else spec.threads
    if callable(grid) or callable(threads):
        meta = {**_named_arguments(argument_names, args, kwargs, spec), **params}
        grid = grid(meta) if callable(grid) else grid
        threads = threads(meta) if callable(threads) else threads
    return grid, threads


class _TuningCallFunction:
    """The call function used by tune_kernel, it launches each configuration like the user launched the kernel.

    The launch dimensions are computed from the copies of the arguments that Kernel Tuner passes. Instances can be
    pickled when the launch dimensions can, so that parallel_compile can compile kernels in worker processes.
    """

    def __init__(self, dsl, argument_names, grid, threads, spec, extra_kwargs):
        self.dsl = dsl
        self.argument_names = argument_names
        self.grid = grid
        self.threads = threads
        self.spec = spec
        self.extra_kwargs = extra_kwargs
        self._states = {}

    def __call__(self, kernel_function, args, kwargs, grid, threads, params):
        grid, threads = _launch_dimensions(
            self.grid, self.threads, self.spec, self.argument_names, args, self.extra_kwargs, params
        )
        state = self._states.setdefault(id(kernel_function), {})
        launch_function = _LAUNCH_FUNCTIONS[self.dsl]
        launch_function(kernel_function, args, {**self.extra_kwargs, **kwargs}, grid, threads, self.spec, params, state)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_states"] = {}  # compiled kernels are not sent to worker processes
        return state


def _configs_to_search_space(configs):
    """Convert a list of configurations to tunable parameters and a restriction that only allows these configurations.

    :param configs: Dictionaries with the values of the tunable parameters, or ``triton.Config`` objects.
    :type configs: list(dict or triton.Config)

    :returns: The tunable parameters and a restriction function.
    :rtype: tuple(dict, callable)
    """
    dicts = []
    for config in configs:
        if isinstance(config, dict):
            dicts.append(dict(config))
        elif hasattr(config, "kwargs") and hasattr(config, "num_warps"):  # triton.Config
            if getattr(config, "pre_hook", None) is not None:
                raise ValueError("triton.Config objects with a pre_hook are not supported")
            values = dict(config.kwargs)
            values["num_warps"] = config.num_warps
            values["num_stages"] = config.num_stages
            # only add the other launch options when they are set, the defaults are left to Triton
            for option, default in (("num_ctas", 1), ("maxnreg", None)):
                if getattr(config, option, default) != default:
                    values[option] = getattr(config, option)
            dicts.append(values)
        else:
            raise TypeError(f"Configurations must be dictionaries or triton.Config objects, got {type(config)}")
    if not dicts:
        raise ValueError("configs must contain at least one configuration")

    names = list(dicts[0])
    for values in dicts:
        for option in ("num_ctas", "maxnreg"):  # Triton options that only some configurations set
            if option in names and option not in values:
                values[option] = {"num_ctas": 1}.get(option)
        if set(values) != set(names):
            raise ValueError(f"All configurations must set the same parameters, got {list(values)} and {names}")

    tune_params = {name: list(dict.fromkeys(values[name] for values in dicts)) for name in names}
    allowed = {tuple(values[name] for name in names) for values in dicts}

    def only_given_configs(params):
        return tuple(params[name] for name in names) in allowed

    return tune_params, only_given_configs


class _Entry:
    """The kernel of the best configuration for a tuning key, with everything needed to launch it."""

    __slots__ = ("kernel", "params", "kwargs", "state")

    def __init__(self, kernel, params, kwargs):
        self.kernel = kernel
        self.params = params
        self.kwargs = kwargs  # the tunable parameters that are kernel arguments
        self.state = {}


class AutotunedKernel:
    """A kernel that is tuned when it is launched for a new tuning key, created by :func:`autotune`."""

    def __init__(self, kernel, tune_params, restrictions, key, grid, threads, reference, atol, wisdom_dir,
                 tune_options):
        """Wrap a kernel decorated by a DSL, see :func:`autotune` for the arguments."""
        self.kernel = kernel
        function = _unwrap(kernel)
        self.name = function.__name__
        self.source_file = inspect.getsourcefile(function)
        self.dsl = detect_python_dsl(self.name, self.source_file)
        if self.dsl not in _LAUNCH_FUNCTIONS:
            raise ValueError(
                f"Could not detect the Python DSL of kernel {self.name}, supported are {list(_LAUNCH_FUNCTIONS)}"
            )
        self._launch_function = _LAUNCH_FUNCTIONS[self.dsl]

        kernel_ast = get_kernel_ast(self.name, self.source_file)
        self.signature = get_arg_names(kernel_ast[1] if isinstance(kernel_ast, tuple) else kernel_ast)

        self.tune_params = tune_params
        self.restrictions = restrictions
        self.key = key
        self.grid = grid
        self.threads = threads
        self.reference = reference
        self.atol = atol
        self.wisdom_dir = wisdom_dir
        self.tune_options = tune_options

        # arguments of the kernel that the user passes, the tunable parameters are passed by Kernel Tuner
        self.argument_names = [name for name in self.signature if name not in tune_params]
        self._dispatch = {}
        self._lock = threading.Lock()
        self.tuning_results = {}  # tuning key -> results of tune_kernel

    @property
    def best_configs(self):
        """The best configuration for each tuning key that the kernel was launched with."""
        return {key: entry.params for key, entry in self._dispatch.items()}

    # launch syntax of the DSLs ---------------------------------------------------------------------------------

    def __getitem__(self, launch):
        """Launch the kernel with a subscript: kernel[grid](*args) or kernel[grid, block](*args)."""
        if self.dsl not in _SUBSCRIPT_DSLS:
            raise TypeError(f"{self.dsl} kernels are launched by calling them, not with kernel[grid](...)")
        if self.dsl in ("numba", "cupyx") and isinstance(launch, tuple) and len(launch) >= 2:
            spec = _Launch(launch[0], launch[1], extra=tuple(launch[2:]))
        else:
            spec = _Launch(grid=launch)
        return lambda *args, **kwargs: self._launch(args, kwargs, spec)

    def __call__(self, *args, **kwargs):
        """Launch the kernel by calling it, or return the script instance or kernel of Tilus and TileLang."""
        if self.dsl == "cupyx":
            grid, block, kernel_args = args
            return self._launch(tuple(kernel_args), {}, _Launch(grid, block, options=kwargs))
        if self.dsl == "tilus":
            if args or kwargs:
                raise TypeError("Tilus scripts with constructor arguments are not supported by autotune")
            return _BoundKernel(self, _Launch())
        if self.dsl == "tilelang":
            # the factory arguments select the kernel, so they are part of the tuning key
            factory = (tuple(args), tuple(sorted(kwargs.items())))
            return _BoundKernel(self, _Launch(options={"factory": factory}))
        if self.dsl in ("taichi", "cute"):
            return self._launch(args, kwargs, _Launch())
        raise TypeError(f"{self.dsl} kernels are launched with kernel[grid](...), see kernel_tuner.decorator")

    def launch(self, *args, **kwargs):
        """Launch the kernel like the launch function of the DSL: ``ct.launch`` for cuTile, ``wp.launch`` for Warp.

        For cuTile: ``kernel.launch(stream, grid, args)``.
        For Warp: ``kernel.launch(dim, inputs, outputs=(), **options)``, the options are passed to ``wp.launch``.
        """
        if self.dsl == "cutile":
            stream, grid, kernel_args = args
            return self._launch(tuple(kernel_args), {}, _Launch(grid=grid, options={"stream": stream}))
        if self.dsl == "warp":
            if len(args) < 1 and "dim" not in kwargs:
                raise TypeError("launch() needs the launch dimensions dim")
            dim = args[0] if args else kwargs.pop("dim")
            inputs = args[1] if len(args) > 1 else kwargs.pop("inputs", ())
            outputs = kwargs.pop("outputs", ())
            return self._launch((*inputs, *outputs), {}, _Launch(grid=dim, options=kwargs))
        raise TypeError(f"launch() is only available for cuTile and Warp kernels, not for {self.dsl}")

    # tuning and dispatch ---------------------------------------------------------------------------------------

    def _tuning_key(self, args, kwargs, spec):
        """Return the tuning key of a launch.

        The key contains the values of the arguments named in key, the data types of all arrays, and the arguments
        of TileLang kernel factories.
        """
        if callable(self.key):
            return (self.key(*args, **kwargs), *spec.key())
        key = []
        if self.key:
            named = _named_arguments(self.argument_names, args, kwargs, spec)
            key.extend(named.get(name) for name in self.key)
        key.extend(_dtype_name(arg.dtype) for arg in (*args, *kwargs.values()) if hasattr(arg, "dtype"))
        return (*key, *spec.key())

    def _launch(self, args, kwargs, spec):
        key = self._tuning_key(args, kwargs, spec)
        entry = self._dispatch.get(key)
        if entry is None:
            with self._lock:
                entry = self._dispatch.get(key)
                if entry is None:
                    entry = self._dispatch[key] = self._select(key, args, kwargs, spec)
        grid, threads = _launch_dimensions(self.grid, self.threads, spec, self.argument_names, args, kwargs,
                                           entry.params)
        return self._launch_function(
            entry.kernel, list(args), {**kwargs, **entry.kwargs}, grid, threads, spec, entry.params, entry.state
        )

    def _select(self, key, args, kwargs, spec):
        """Select the best configuration for the tuning key from the wisdom file, or by tuning the kernel."""
        import torch

        device_name = torch.cuda.get_device_name()
        tunable_parameters = list(self.tune_params)
        params = None
        filename = wisdom.wisdom_file(self.wisdom_dir, self.name) if self.wisdom_dir else None
        if filename:
            records = wisdom.read_wisdom(filename, tunable_parameters)
            params = wisdom.best_wisdom_config(records, key, device_name, tunable_parameters)
        if params is None:
            results = self._tune(args, kwargs, spec)
            self.tuning_results[key] = results
            valid = [result for result in results if isinstance(result.get("time"), float)]
            if not valid:
                raise RuntimeError(f"No valid configuration found while tuning {self.name} for key {key}")
            best = min(valid, key=lambda result: result["time"])
            params = {name: best[name] for name in tunable_parameters}
            if filename:
                wisdom.write_wisdom(filename, self.name, tunable_parameters, key, device_name, valid)
        return self._entry(params)

    def _entry(self, params):
        """Create the kernel of a configuration, in the same way as Kernel Tuner does while tuning."""
        kernel_source = KernelSourceFn(self.name, self.source_file, "generic_python", call_function=_not_called)
        source_params = {name: value for name, value in params.items() if name not in self.signature}
        kernel, temp_file = kernel_source.apply_params_to_source_fn(source_params)
        # DSLs may read the source file of a kernel again when they compile it for new argument types
        atexit.register(delete_temp_file, temp_file)
        kwargs = {name: params[name] for name in self.signature if name in params}
        return _Entry(kernel, params, kwargs)

    def _answer(self, args, kwargs, kernel_args):
        """Return the expected outputs computed by the reference function, in the form of tune_kernel's answer."""
        if self.reference is None:
            return None
        expected = self.reference(*args, **kwargs)
        if isinstance(expected, dict):
            if self.dsl == "tilelang":
                raise ValueError("reference must return a list for TileLang kernels, the arguments have no names")
            names = self.argument_names
            unknown = [name for name in expected if name not in names[: len(kernel_args)]]
            if unknown:
                raise ValueError(f"reference returned outputs for unknown arguments {unknown}, arguments are {names}")
            answer = [expected.get(name) for name in names[: len(kernel_args)]]
        else:
            answer = list(expected) + [None] * (len(kernel_args) - len(expected))
        return [None if value is None else _expected_like(value, arg) for value, arg in zip(answer, kernel_args)]

    def _tune(self, args, kwargs, spec):
        """Tune the kernel with tune_kernel on copies of the arguments."""
        from kernel_tuner.interface import tune_kernel

        # kernel arguments passed by name are passed in the order of the signature
        kernel_args = list(args)
        for name in self.argument_names[len(args):]:
            if name not in kwargs:
                break
            kernel_args.append(kwargs[name])
        extra_kwargs = {name: value for name, value in kwargs.items() if name not in self.argument_names}
        # streams cannot be sent to worker processes, configurations are benchmarked on the current stream
        launch_options = {name: value for name, value in spec.options.items() if name != "stream"}
        extra = (0, *spec.extra[1:]) if spec.extra else ()
        tuning_spec = _Launch(spec.grid, spec.threads, extra, launch_options)
        call_function = _TuningCallFunction(
            self.dsl, self.argument_names, self.grid, self.threads, tuning_spec, extra_kwargs
        )

        kernel_args = [_to_torch(arg) for arg in kernel_args]
        options = dict(self.tune_options)
        options.setdefault("strategy", "brute_force")
        options.setdefault("quiet", True)
        restrictions = options.pop("restrictions", self.restrictions)
        with warnings.catch_warnings():
            # the launch dimensions are computed by the call function, not from block size parameters
            warnings.simplefilter("ignore", UserWarning)
            results, _ = tune_kernel(
                self.name,
                self.source_file,
                1,
                kernel_args,
                self.tune_params,
                restrictions=restrictions,
                lang="generic_python",
                call_function=call_function,
                answer=self._answer(args, kwargs, kernel_args),
                atol=self.atol,
                **options,
            )
        logging.debug(f"tuned {self.name}: {len(results)} configurations")
        return results


class _BoundKernel:
    """A Tilus script instance or a TileLang kernel returned by a factory, launched by calling it."""

    def __init__(self, autotuned, spec):
        self._autotuned = autotuned
        self._spec = spec

    def __call__(self, *args, **kwargs):
        return self._autotuned._launch(args, kwargs, self._spec)


def _not_called(kernel_function, args, kwargs):
    """Kernels created for dispatch are launched by the launch functions of the decorator."""
    raise RuntimeError("not called")


def autotune(tune_params=None, *, configs=None, restrictions=None, key=None, grid=None, threads=None,
             reference=None, atol=1e-6, wisdom=None, **tune_options):
    """Tune a kernel written in a Python-embedded DSL when it is launched for a new tuning key.

    :param tune_params: The tunable parameters and their values, as in tune_kernel.
    :type tune_params: dict(str: list)

    :param configs: Instead of tune_params, a list of configurations: dictionaries with the values of the tunable
        parameters or ``triton.Config`` objects. Only these configurations are benchmarked.
    :type configs: list(dict or triton.Config)

    :param restrictions: Restrictions on the tunable parameters, as in tune_kernel.
    :type restrictions: list(str) or callable

    :param key: The names of the kernel arguments whose values are part of the tuning key, or a function of the
        kernel arguments that returns the key. The data types of the array arguments are always part of the key.
        When the kernel is launched with a new key, the kernel is tuned again.
    :type key: list(str) or callable

    :param grid: The grid dimensions, or a function of a dictionary with the kernel arguments and tunable
        parameters that returns the grid dimensions. Overrides the grid of the launch.
    :type grid: tuple or callable

    :param threads: The thread block dimensions, or a function like grid. Overrides the block of the launch.
    :type threads: tuple or callable

    :param reference: A function of the kernel arguments that returns the expected outputs, as a dictionary with
        the names of the output arguments, or as a list with one value per argument (None for inputs). The outputs
        of every configuration are compared with these while tuning. By default the outputs are not verified.
    :type reference: callable

    :param atol: The tolerance used to compare the outputs with those of the reference function.
    :type atol: float

    :param wisdom: A directory in which the tuning results are stored, in wisdom files named after the kernel.
        Configurations stored for the same tuning key and GPU are used instead of tuning the kernel again.
    :type wisdom: str

    :param tune_options: Other options passed to tune_kernel, such as strategy, strategy_options, iterations,
        parallel_compile, or verbose. The strategy is brute_force by default.

    :returns: A decorator that returns an AutotunedKernel.
    """
    if (tune_params is None) == (configs is None):
        raise ValueError("Pass either tune_params or configs to autotune")
    if configs is not None:
        tune_params, only_given_configs = _configs_to_search_space(configs)
        if restrictions is None:
            restrictions = [only_given_configs]
        elif callable(restrictions):
            restrictions = [restrictions, only_given_configs]
        else:
            restrictions = [*restrictions, only_given_configs]
    tune_options.setdefault("quiet", not tune_options.get("verbose", False))
    wisdom_dir = os.fspath(wisdom) if wisdom is not None else None

    def decorator(kernel):
        return AutotunedKernel(kernel, tune_params, restrictions, key, grid, threads, reference, atol, wisdom_dir,
                               tune_options)

    return decorator
