import kernel_tuner

import numpy as np
import pytest


def test_tunable():
    from kernel_tuner.accuracy import Tunable

    # Test with string as key
    x = Tunable("foo", dict(a=1, b=2))
    assert x(dict(foo="a")) == 1
    assert x(dict(foo="b")) == 2

    with pytest.raises(KeyError):
        assert x(dict(foo="c")) == 3

    # Test with lambda as key
    x = Tunable(lambda p: p["foo"] + p["bar"], dict(ab=1, bc=2))
    assert x(dict(foo="a", bar="b")) == 1
    assert x(dict(foo="b", bar="c")) == 2

    with pytest.raises(KeyError):
        assert x(dict(foo="c", bar="d")) == 3


def test_to_float_dtype():
    from kernel_tuner.accuracy import _to_float_dtype

    ## Unfortunately, numpy does not offer bfloat16
    # assert _to_float_dtype("bfloat16") == np.bfloat16

    assert _to_float_dtype("half") == np.float16
    assert _to_float_dtype("f16") == np.float16
    assert _to_float_dtype("float16") == np.float16

    assert _to_float_dtype("float") == np.float32
    assert _to_float_dtype("f32") == np.float32
    assert _to_float_dtype("float32") == np.float32

    assert _to_float_dtype("double") == np.float64
    assert _to_float_dtype("f64") == np.float64
    assert _to_float_dtype("float64") == np.float64


def test_tunable_precision():
    from kernel_tuner.accuracy import TunablePrecision

    inputs = np.array([1, 2, 3], dtype=np.float64)
    x = TunablePrecision(
        "foo", inputs, dict(float16=np.half, float32=np.float32, float64=np.double)
    )

    assert np.all(x(dict(foo="float16")) == inputs)
    assert x(dict(foo="float16")).dtype == np.half

    assert np.all(x(dict(foo="float32")) == inputs)
    assert x(dict(foo="float32")).dtype == np.float32

    assert np.all(x(dict(foo="float64")) == inputs)
    assert x(dict(foo="float64")).dtype == np.double


def test_error_metric_from_name():
    from kernel_tuner.accuracy import error_metric_from_name
    from math import sqrt

    eps = 0.1
    a = np.array([0, 1, 2, 3])
    b = np.array([1, 1, 2, 5])

    assert error_metric_from_name("mse")(a, b) == pytest.approx(1.25)
    assert error_metric_from_name("rmse")(a, b) == pytest.approx(sqrt(1.25))
    assert error_metric_from_name("nrmse")(a, b) == pytest.approx(sqrt(1.25 / 3.5))
    assert error_metric_from_name("mae")(a, b) == pytest.approx(0.75)
    assert error_metric_from_name("mre", eps)(a, b) == pytest.approx(2.666666666666666)
    assert error_metric_from_name("rmsre", eps)(a, b) == pytest.approx(5.011098792790969)
    assert error_metric_from_name("male", eps)(a, b) == pytest.approx(0.3144002918554722)
    assert error_metric_from_name("rmsle", eps)(a, b) == pytest.approx(0.5317999700319226)
    assert error_metric_from_name("maximum abs")(a, b) == pytest.approx(2)
    assert error_metric_from_name("maximum rel", eps)(a, b) == pytest.approx(10)
