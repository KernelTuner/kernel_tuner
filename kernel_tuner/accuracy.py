from collections import UserDict
from typing import Dict
import numpy as np

from kernel_tuner.observers import AccuracyObserver


class Tunable(UserDict):
    def __init__(self, param_key: str, arrays: Dict):
        """The ``Tunable`` object is used as an input argument when tuning
        kernels. It is a container that holds several arrays internally and
        selects one array during benchmarking based on a tunable parameter.

        Example
        -------
        Consider this example::

            arg = Tunable("matrix_layout", dict("c"=matrix, "f"=matrix.transpose()))

        In this example, we create a Tunable object that selects either matrix
        or matrix.transpose() for benchmarking, depending on the value of the
        tunable parameter "matrix_layout". The arrays argument is a dictionary
        that maps the tunable parameter values "c" and "f" to the arrays matrix
        and matrix.transpose(), respectively. During benchmarking, the Tunable
        object selects the appropriate array based on the value of "matrix_layout".

        :param param_key: : The tunable parameter used to select the array for benchmarking.
        :param arrays: A dictionary that maps the parameter value to arrays.
        """
        if isinstance(arrays, (tuple, list)):
            arrays = dict(enumerate(arrays))

        super().__init__(arrays)
        self.param_key = param_key

    def select_for_configuration(self, params):
        if callable(self.param_key):
            key = self.param_key(params)
        elif self.param_key in params:
            key = params[self.param_key]
        else:
            key = eval(self.param_key, params, params)

        if key not in self:
            list = ", ".join(map(str, self.keys()))
            raise KeyError(
                f"'{key}' is not a valid parameter value, should be one of: {list}"
            )

        return self[key]

    def __call__(self, params):
        return self.select_for_configuration(params)


def _to_float_dtype(x):
    """Convert a string to a numpy data type (``dtype``). This function recognizes
    common names (such as ``f16`` or ``kfloat``), and uses ``np.dtype(x)`` as a
    fallback.
    """
    if isinstance(x, str):
        x = x.lower()

    if x in ("bfloat16", "bf16", "kbfloat16", "__nv_bfloat16"):
        from bfloat16 import bfloat16

        return bfloat16
    if x in ("half", "f16", "float16", "__half", "khalf", 16):
        return np.half
    if x in ("float", "single", "f32", "float32", "kfloat", 32):
        return np.float32
    if x in ("double", "f64", "float64", "kdouble", 64):
        return np.float64

    return np.dtype(x)


class TunablePrecision(Tunable):
    def __init__(
        self, param_key: str, array: np.ndarray, dtypes: Dict[str, np.dtype] = None
    ):
        """The ``Tunable`` object is used as an input argument when tuning
        kernels. It is a container that internally holds several arrays
        containing the same data, but stored in using different levels of
        precision. During benchamrking, one array is selected based on a
        tunable parameter ``param_key``.

        Example
        -------
        Consider this example::

            arg = TunablePrecision("matrix_type", matrix)

        This creates a ``TunablePrecision`` argument that selects the required
        floating-point precision for ``matrix`` based on the tunable parameter
        ``"matrix_type"``.

        :param param_key: The tunable parameter used to select the level of precision.
        :param array: The input array. It will automatically be converted to
                      all data types given by ``dtypes``.
        :param dtypes: Dictionary that maps names to numpy data types. The default
                       types are ``double``, ``float``, and ``half``.
        """
        # If no dtypes are given, generate a default list
        if not dtypes:
            dtypes = dict(half=np.half, float=np.single, double=np.double)

            # Try to get bfloat16 if available.
            try:
                from bfloat16 import bfloat16

                dtypes["bfloat16"] = bfloat16
                pass
            except ImportError:
                pass  # Ignore error if tensorflow is not available

        # If dtype is a list, convert it to a dictionary
        if isinstance(dtypes, (list, tuple)):
            dtypes = dict((name, _to_float_dtype(name)) for name in dtypes)

        arrays = dict()
        for precision, dtype in dtypes.items():
            # We convert the array into a `np.ndarray` by using `np.array`.
            # However, if the value is a numpy scalar, then we do not want to
            # convert it into an array but instead keep the original value
            if not isinstance(array, np.generic):
                array = np.array(array)

            arrays[precision] = array.astype(dtype)

        super().__init__(param_key, arrays)


class ErrorMetricObserver(AccuracyObserver):
    """An ``AccuracyObserver`` that measure the error of the outputs produced
    by a kernel by comparing it against reference outputs.

    By default, it uses the mean-squared error (MSE) and appends this to
    the results with a metric called ``error``.
    """

    def __init__(self, metric=None, key="error"):
        """Create a new ``AccuracyObserver``.

        :param metric: The error metric. Should be function that accepts two numpy
                       arrays as arguments (the reference output and the kernel output)
        :param key: The name of this metric in the results.
        """

        # The default metric is the mean squared error
        if metric is None:
            metric = lambda a, b: np.average(np.square(a - b))

        self.key = key
        self.metric = metric
        self.result = None

    def process_kernel_output(self, answers, outputs):
        errors = []

        for answer, output in zip(answers, outputs):
            if answer is not None:
                errors.append(self.metric(answer, output))

        self.result = max(errors)

    def get_results(self):
        return dict([(self.key, self.result)])
