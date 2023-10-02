from collections import UserDict
from typing import Dict
import numpy as np
import logging
import re

from kernel_tuner.observers import OutputObserver


class Tunable(UserDict):
    def __init__(self, param_key: str, arrays: Dict):
        """The ``Tunable`` object can be used as an input argument when tuning
        kernels. It is a container that holds several arrays internally and
        selects one array during benchmarking based on the value of a tunable parameter.

        Example
        -------
        Consider this example::

            arg = Tunable("matrix_layout", dict("c"=matrix, "f"=matrix.transpose()))

        In this example, we create a Tunable object that selects either matrix
        or matrix.transpose() for benchmarking, depending on the value of the
        tunable parameter "matrix_layout". The first argument is the name of the tunable
        paramater. The second argument is a dictionary that maps the tunable parameter
        values "c" and "f" to the arrays ``matrix`` and ``matrix.transpose()``, respectively.
        During benchmarking, the Tunable object selects the array passed to the kernel based
        on the value of "matrix_layout".

        :param param_key: : The tunable parameter used to select the array for benchmarking.
        :param arrays: A dictionary that maps the value of that tunable parameter to options.
        """
        if isinstance(arrays, (tuple, list)):
            arrays = dict(enumerate(arrays))

        super().__init__(arrays)
        self.param_key = param_key

    def select_for_configuration(self, params):
        if callable(self.param_key):
            option = self.param_key(params)
        elif self.param_key in params:
            option = params[self.param_key]
        else:
            option = eval(self.param_key, params, params)

        if option not in self.data:
            list = ", ".join(map(str, self.data.keys()))
            raise KeyError(
                f"'{option}' is not a valid parameter value, should be one of: {list}"
            )

        return self.data[option]

    def __call__(self, params):
        return self.select_for_configuration(params)


def _find_bfloat16_if_available():
    # Try to get bfloat16 if available.
    try:
        from bfloat16 import bfloat16
        return bfloat16
    except ImportError:
        pass

    try:
        from tensorflow import bfloat16
        return bfloat16.as_numpy_dtype
    except ImportError:
        pass

    logging.warning(
        "could not find `bfloat16` data type for numpy, "
        + "please install either the package `bfloat16` or `tensorflow`"
    )

    return None


def _to_float_dtype(x: str) -> np.dtype:
    """Convert a string to a numpy data type (``dtype``). This function recognizes
    common names (such as ``f16`` or ``kfloat``), and uses ``np.dtype(x)`` as a
    fallback.
    """
    if isinstance(x, str):
        x = x.lower()

    if x in ("bfloat16", "bf16", "kbfloat16", "__nv_bfloat16"):
        result = _find_bfloat16_if_available()
        if result is not None:
            return result

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
        """The ``Tunable`` object can be used as an input argument when tuning
        kernels. It is a container that internally holds several arrays
        containing the same data, but stored in using different levels of
        precision. During benchamrking, one array is selected based on the value
        of the tunable parameter called ``param_key``.

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

            bfloat16 = _find_bfloat16_if_available()
            if bfloat16 is not None:
                dtypes["bfloat16"] = bfloat16


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


def error_metric_from_name(user_key, EPS=1e-8):
    """Find the error metric function for the given name.

    Returns an function that takes two parameters (the ground-truth and the
    estimated values) as numpy arrays and returns the error between the two
    according to the given error metric.

    Valid values for the ``key`` are:

    * MSE (mean square error)
    * RSME (Root mean square error)
    * NRMSE (normalized root mean square error)
    * RMSRE (root mean square relative error)
    * RMSLE (root mean square log error)
    * MAE (mean absolute error)
    * MRE (mean relative error)
    * MALE (mean absolute log error)
    * max (maximum absolute error)
    * max rel (maximum relative error)

    The value of `EPS` is used for relative errors to prevent division by zero.
    ``
    """

    # Prepocess the provided name:
    # - convert to lowercase
    # - remove the word "error"
    # - remove underscores and dashes
    # - strip whitespaces
    # - replace common abreviations
    key = user_key.lower()
    key = re.sub(r"\berror\b", " ", key)
    key = re.sub(r"[\s_-]+", " ", key)
    key = key.strip()

    replacements = {
        "average": "mean",
        "avg": "mean",
        "square": "squared",
        "sq": "squared",
        "max": "maximum",
        "rel": "relative",
        "abs": "absolute",
        "log": "logarithmic",
    }

    for pattern, replacement in replacements.items():
        key = re.sub(rf"\b{pattern}\b", replacement, key)

    # Select the right metric
    if key in ("mse", "mean squared"):

        def metric(a, b):
            return np.average(np.square(a - b))

    elif key in ("rmse", "root mean squared"):

        def metric(a, b):
            return np.sqrt(np.average(np.square(a - b)))

    elif key in ("nrmse", "normalized root mean squared"):

        def metric(a, b):
            return np.sqrt(np.average(np.square(a - b)) / np.average(np.square(a)))

    elif key in ("mae", "absolute", "mean absolute"):

        def metric(a, b):
            return np.average(np.abs(a - b))

    elif key in ("mre", "relative", "mean relative"):

        def metric(a, b):
            return np.average(np.abs(a - b) / np.maximum(np.abs(a), EPS))

    elif key in ("rmsre", "root mean squared relative"):

        def metric(a, b):
            return np.sqrt(np.average(np.square(a - b) / np.square(np.maximum(a, EPS))))

    elif key in ("male", "mean absolute logarithmic"):

        def metric(a, b):
            return np.average(np.abs(np.log10(a + EPS) - np.log10(b + EPS)))

    elif key in ("rmsle", "root mean squared logarithmic"):

        def metric(a, b):
            return np.sqrt(np.average(np.square(np.log10(a + EPS) - np.log10(b + EPS))))

    elif key in ("maximum absolute", "maximum"):

        def metric(a, b):
            return np.amax(np.abs(a - b))

    elif key in ("maximum relative",):

        def metric(a, b):
            return np.amax(np.abs(a - b) / np.maximum(np.abs(a), EPS))

    else:
        raise ValueError(f"invalid error metric provided: {user_key}")

    # cast both arguments to f64 before passing them to the metric
    return lambda a, b: metric(
        a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
    )


class AccuracyObserver(OutputObserver):
    """``AccuracyObserver`` measures the error on the output produced by a kernel
    by comparing the output against a reference output.

    By default, it uses the root mean-squared error (RMSE) and uses the
    metric name ``"error"``.
    """

    def __init__(self, metric=None, key="error", *, atol=1e-8):
        """Create a new ``AccuracyObserver``.

        :param metric: The error metric. This should be a string that is
                       accepted by ``error_metric_from_name`` such as ``"absolute error"``
                       or ``"relative error"``. Alternatively, it can be a
                       function that accepts two numpy arrays as arguments
                       (the reference output and the kernel output)
        :param key: The name of this metric in the results.
        :param atol: The tolerance used in relative metrics to prevent
                     division by zero. It is ignored by absolute error metrics.
        """

        # Default metric is RMSE
        if not metric:
            metric = "rmse"

        # If it is a string, convert it to a function
        if isinstance(metric, str):
            metric = error_metric_from_name(metric, atol)

        self.key = key
        self.metric = metric
        self.result = None

    def process_output(self, answers, outputs):
        errors = []

        for answer, output in zip(answers, outputs):
            if answer is not None:
                errors.append(self.metric(answer, output))

        self.result = max(errors)

    def get_results(self):
        return dict([(self.key, self.result)])
