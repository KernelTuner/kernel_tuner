
import kernel_tuner
from .context import skip_if_no_cupy
from .test_runners import env

@skip_if_no_cupy
def test_tune_kernel(env):
    result, _ = kernel_tuner.tune_kernel(*env, lang="cupy", verbose=True)
    assert len(result) > 0

