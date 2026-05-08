import json
import subprocess
import sys


def _run_isolated(code):
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_import_kernel_tuner_does_not_import_optional_deps():
    optional_modules = {"pycuda", "pyopencl", "cupy", "hip", "skopt", "sklearn", "pyatf"}
    code = f"""
import json, sys
optional = {sorted(optional_modules)!r}
import kernel_tuner  # noqa: F401
loaded = sorted(set(optional) & set(sys.modules))
print(json.dumps(loaded))
"""
    loaded = json.loads(_run_isolated(code))
    assert loaded == []


def test_strategy_modules_loaded_on_demand():
    code = """
import json, sys
import kernel_tuner.interface as interface
before = "kernel_tuner.strategies.brute_force" in sys.modules
_ = interface.strategy_map["brute_force"].tune
after = "kernel_tuner.strategies.brute_force" in sys.modules
print(json.dumps({"before": before, "after": after}))
"""
    result = json.loads(_run_isolated(code))
    assert result["before"] is False
    assert result["after"] is True


def test_backend_modules_not_loaded_on_import():
    code = """
import json, sys
import kernel_tuner.core as core  # noqa: F401
mods = ["kernel_tuner.backends.pycuda", "kernel_tuner.backends.opencl"]
loaded = [m for m in mods if m in sys.modules]
print(json.dumps(loaded))
"""
    loaded = json.loads(_run_isolated(code))
    assert loaded == []
