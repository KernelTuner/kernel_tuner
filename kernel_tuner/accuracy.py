from collections import UserDict
from typing import Dict
import numpy as np

from kernel_tuner.observers import AccuracyObserver


class Tunable(UserDict):
    def __init__(self, param_key: str, arrays: Dict):
        """Create a new ``Tunable``.

        ``Tunable`` can be used one of the input arguments when tuning kernels. It can contain
        several arrays and the array that will be used during benchmark of one kernel configuration
        can be selected based on a tunable parameter.

        Example
        -------
        For example, it is possible to define a tunable parameter called ``matrix_layout`` and then
        tunable for Fortran-order or C-order memory layout by passing the following object as a
        kernel argument:

        ```
        Tunable("matrix_layout", dict("c"=matrix, "f"=matrix.transpose()))
        ```

        :param param_key: The tunable parameter used to select the array for benchmarking.
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
            raise KeyError(f"'{key}' is not a valid parameter value, should be one of: {list}")

        return self[key]

    def __call__(self, params):
        return self.select_for_configuration(params)


class TunablePrecision(Tunable):
    def __init__(self, param_key: str, array: np.ndarray, dtypes: Dict[str, np.dtype] = None):
        """
        Create a new ``TunablePrecision``.

        ``TunablePrecision`` can be used one of the input arguments when tuning kernels. It
        contains the same array data, but stored using different levels of precision. This can
        be used to tune the optimal precision for a kernel argument.

        :param param_key: The tunable parameter used to select the precision for benchmarking.
        :param array: The input array. Will be converted to the given precision levels.
        :param dtypes: Dictionary that maps names to numpy data types.
        """
        # If no dtypes are given, generate a default list
        if not dtypes:
            dtypes = dict(
                half=np.half,
                float=np.single,
                double=np.double)

            # Try to get bfloat16 from tensorflow if available.
            try:
                #import tensorflow
                #dtypes["bfloat16"] = tensorflow.bfloat16.as_numpy_dtype
                pass
            except ImportError:
                pass  # Ignore error if tensorflow is not available

        # If dtype is a list, convert it to a dictionary
        if isinstance(dtypes, (list, tuple)):
            dtypes = dict((name, np.dtype(name)) for name in dtypes)

        arrays = dict()
        for precision, dtype in dtypes.items():
            arrays[precision] = np.array(array).astype(dtype)

        super().__init__(param_key, arrays)


class CalculateErrorObserver(AccuracyObserver):
    def __init__(self, metric=None, key="error"):
        # The default metric is the mean squared error
        if metric is None:
            metric = lambda a, b: np.average(np.square(a - b))

        self.key = key
        self.function = metric
        self.result = None

    def process_kernel_output(self, answers, outputs):
        errors = []

        for answer, output in zip(answers, outputs):
            if answer is not None:
                errors.append(self.metric(answer, output))

        self.result = max(errors)

    def get_results(self):
        return dict([(self.key, self.result)])
