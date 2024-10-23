from kernel_tuner.integration import store_results, create_device_targets
from kernel_tuner.interface import tune_kernel, tune_kernel_T1, run_kernel

from importlib.metadata import version

__version__ = version(__package__)
