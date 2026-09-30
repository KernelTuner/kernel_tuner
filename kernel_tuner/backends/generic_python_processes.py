"""Compile Python DSL kernels in worker processes, for DSLs that cannot compile kernels in parallel threads.

Numba, Warp, and cuTile hold a global lock while compiling, and CuTe kernels can only be launched from the
thread that compiled them. However, these DSLs can store compiled kernels on disk. A worker process imports
the same kernel module file as the main process, and compiles the kernel by running it once, after which the
main process loads the compiled kernel from disk instead of compiling it:

- Numba: caching is enabled on the kernel, the cache files are removed once the main process loaded them.
- Warp and cuTile: their kernel caches are enabled by default.
- CuTe: ``cute.compile`` does not use the file cache of CuTe, so the compiled kernel is exported with
  ``export_to_c`` and loaded with ``cutlass.runtime.load_module``.

If a kernel cannot be compiled in a worker process, for example because the call function cannot be sent to
the worker, the main process compiles the kernel instead.
"""

import glob
import importlib.util
import inspect
import itertools
import logging
import multiprocessing
import os
import pickle
import shutil
import sys
import tempfile
import threading
import weakref
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager

from kernel_tuner.kernel_sources.kernel_source_fn import kernel_module_name

# DSLs whose kernels are compiled in worker processes when compiling in parallel
PROCESS_BUILD_DSLS = {"numba", "warp", "cutile", "cute"}

BuildJob = namedtuple(
    "BuildJob",
    ["module_path", "module_name", "kernel_name", "dsl", "call_function", "args", "kwargs", "grid", "threads", "params",
     "export_dir"],
)


@contextmanager
def dsl_compile_hooks(dsl, kernel_function, module_name, export_dir, export):
    """Store the kernel compiled while running the call function on disk, or load it from disk.

    :param export: True in a worker process, to store the compiled kernel on disk. False in the main process,
        to load the kernel stored by the worker process.
    :type export: bool
    """
    if dsl == "numba" and hasattr(kernel_function, "enable_caching"):
        kernel_function.enable_caching()
        yield
        if not export:
            _remove_numba_cache_files(kernel_function)
    elif dsl == "cute":
        with _cute_export_hook(module_name, export_dir, export):
            yield
    else:
        yield


def _remove_numba_cache_files(kernel_function):
    """Remove the cache files of a Numba kernel, once it is loaded they are no longer needed."""
    try:
        cache = kernel_function._cache
        for path in glob.glob(os.path.join(cache.cache_path, glob.escape(cache._impl.filename_base) + ".*")):
            os.remove(path)
    except Exception as e:
        logging.debug(f"could not remove Numba cache files: {e}")


@contextmanager
def _cute_export_hook(module_name, export_dir, export):
    """Replace cute.compile to export compiled kernels, or to load exported kernels instead of compiling them."""
    import cutlass.cute as cute
    import cutlass.runtime

    original_compile = cute.compile
    counter = itertools.count()
    keys = []

    def compile_hook(*args, **kwargs):
        # call functions compile the kernels in the same order in the worker and in the main process
        key = f"{module_name}_{next(counter)}"
        keys.append(key)
        object_file = os.path.join(export_dir, key + ".o")
        if not export and os.path.exists(object_file):
            try:
                return cutlass.runtime.load_module(object_file)[key]
            except Exception as e:
                logging.debug(f"could not load exported CuTe kernel {object_file}, compiling it instead: {e}")
        compiled = original_compile(*args, **kwargs)
        if export:
            try:
                compiled.export_to_c(export_dir, key, function_prefix=key)
            except Exception as e:
                logging.debug(f"could not export CuTe kernel {key}: {e}")
        return compiled

    cute.compile = compile_hook
    try:
        yield
    finally:
        cute.compile = original_compile
        if not export:
            for key in keys:
                for path in glob.glob(os.path.join(export_dir, glob.escape(key) + ".*")):
                    os.remove(path)


def instantiate_kernel(kernel_fn):
    """Class-based kernels are instantiated, their __call__ method is the kernel."""
    if inspect.isclass(kernel_fn):
        return kernel_fn()
    if callable(kernel_fn):
        return kernel_fn
    raise TypeError("kernel function is not a class or function")


def _to_host(arg):
    """Copy PyTorch GPU tensors to the host, to send them to a worker process."""
    import torch

    if isinstance(arg, torch.Tensor) and arg.is_cuda:
        return arg.cpu()
    return arg


def _to_device(arg):
    import torch

    if isinstance(arg, torch.Tensor):
        return arg.to("cuda")
    return arg


def _picklable_exception_chain(e):
    """Return the exception and the exceptions it was raised from, as they can be sent to the main process.

    The chain is followed like Python does when printing a traceback. Exceptions that cannot be pickled are
    replaced by a RuntimeError with the name of the exception type and its message.
    """
    chain = []
    while e is not None and all(e is not seen for seen in chain):
        chain.append(e)
        e = e.__cause__ if e.__cause__ is not None else (None if e.__suppress_context__ else e.__context__)
    picklable = []
    for exc in chain:
        try:
            pickle.loads(pickle.dumps(exc))
            picklable.append(exc)
        except Exception:
            picklable.append(RuntimeError(f"{type(exc).__name__}: {exc}"))
    return picklable


def _raise_exception_chain(chain):
    for exc, cause in zip(chain, chain[1:]):
        exc.__cause__ = cause
    raise chain[0]


# state of a worker process, set by _init_worker
_worker = {}


def _init_worker(device_id, args):
    import torch

    torch.cuda.set_device(device_id)
    _worker["args"] = [_to_device(arg) for arg in args]
    _worker["broken"] = False


def _cuda_is_healthy():
    import torch

    try:
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _build_in_worker(job):
    """Compile a kernel in a worker process by running it once.

    :returns: None if the kernel was compiled, or the exception chain of the error raised while compiling.
    """
    import torch

    if _worker["broken"]:
        # the CUDA context of this process is unusable after a sticky error, such as an illegal memory access,
        # exit and let the main process start a new worker
        os._exit(1)

    try:
        args = list(_worker["args"])
        for index, arg in job.args.items():
            args[index] = _to_device(arg)

        spec = importlib.util.spec_from_file_location(job.module_name, job.module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[job.module_name] = module
        spec.loader.exec_module(module)
        kernel_function = instantiate_kernel(getattr(module, job.kernel_name))

        with dsl_compile_hooks(job.dsl, kernel_function, job.module_name, job.export_dir, export=True):
            torch.cuda.synchronize()
            job.call_function(kernel_function, args, job.kwargs, job.grid, job.threads, job.params)
            torch.cuda.synchronize()
        return None
    except Exception as e:
        _worker["broken"] = not _cuda_is_healthy()
        return _picklable_exception_chain(e)
    finally:
        sys.modules.pop(job.module_name, None)


class BuildProcessPool:
    """A pool of worker processes that compile kernels, so that the main process can load them from disk."""

    def __init__(self, device_id, gpu_args, num_processes):
        """Create the pool, the worker processes are started when kernels are built.

        :param gpu_args: The kernel arguments, these are copied to each worker process once.
        :type gpu_args: list

        :param num_processes: The maximum number of worker processes.
        :type num_processes: int
        """
        self.device_id = device_id
        self.gpu_args = list(gpu_args)
        self.num_processes = num_processes
        self.export_dir = tempfile.mkdtemp(prefix="kernel_tuner_build_")
        # also remove the exported kernels when tuning stops with an error, and shutdown() is not called
        weakref.finalize(self, shutil.rmtree, self.export_dir, ignore_errors=True)
        self.disabled = False
        self.any_succeeded = False
        self._executor = None
        self._lock = threading.Lock()

    def _get_executor(self):
        with self._lock:
            if self._executor is None:
                # CUDA cannot be used in forked processes
                self._executor = ProcessPoolExecutor(
                    max_workers=self.num_processes,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(self.device_id, [_to_host(arg) for arg in self.gpu_args]),
                )
            return self._executor

    def build(self, kernel_instance, gpu_args, call_function, kwargs):
        """Compile the kernel of kernel_instance in a worker process, this is thread-safe.

        Errors raised while compiling the kernel are raised again in this process.

        :returns: True if the kernel was compiled by a worker process, False if it must be compiled by the
            main process instead.
        :rtype: bool
        """
        if self.disabled:
            return False

        # only send the arguments that differ from the arguments the workers already have, such as
        # scalars and arguments that depend on the configuration
        args = {i: _to_host(arg) for i, arg in enumerate(gpu_args) if arg is not self.gpu_args[i]}
        job = BuildJob(
            module_path=kernel_instance.temp_files[0],
            module_name=kernel_module_name(kernel_instance.temp_files[0]),
            kernel_name=kernel_instance.name,
            dsl=kernel_instance.kernel_source.dsl,
            call_function=call_function,
            args=args,
            kwargs=kwargs,
            grid=kernel_instance.grid,
            threads=kernel_instance.threads,
            params=kernel_instance.params,
            export_dir=self.export_dir,
        )

        executor = self._get_executor()
        try:
            error_chain = executor.submit(_build_in_worker, job).result()
        except BrokenProcessPool as e:
            self._handle_broken_pool(executor, e)
            return False
        except Exception as e:
            # usually the job could not be pickled, for example because the call function is a lambda
            logging.warning(f"Could not compile kernels in worker processes, compiling in this process instead: {e}")
            self.disabled = True
            return False

        if error_chain is not None:
            _raise_exception_chain(error_chain)
        self.any_succeeded = True
        return True

    def _handle_broken_pool(self, executor, error):
        """Start new worker processes after a worker exited, unless worker processes never worked at all."""
        with self._lock:
            if not self.any_succeeded:
                # for example, the worker processes cannot import the script that calls tune_kernel
                logging.warning(
                    f"Could not compile kernels in worker processes, compiling in this process instead: {error}"
                )
                self.disabled = True
            if self._executor is executor:
                self._executor = None
        executor.shutdown(wait=False, cancel_futures=True)

    def shutdown(self):
        """Stop the worker processes and remove the exported kernels."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(self.export_dir, ignore_errors=True)
