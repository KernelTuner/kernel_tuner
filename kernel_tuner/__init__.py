from importlib.metadata import version

from kernel_tuner.interface import run_kernel, tune_kernel, tune_kernel_T1

__version__ = version(__package__)

__all__ = [
    "create_device_targets",
    "run_kernel",
    "store_results",
    "tune_kernel",
    "tune_kernel_T1",
    "__version__",
]


def __getattr__(name):
    if name in ("store_results", "create_device_targets"):
        from kernel_tuner import integration
        return getattr(integration, name)
    raise AttributeError(f"module 'kernel_tuner' has no attribute {name!r}")
