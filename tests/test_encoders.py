from contextlib import contextmanager
from io import BytesIO

import numpy as np
import pytest

from gribgen.encoders import SimplePackingEncoder


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "data,r,e,d,expected",
    [
        (
            np.array([0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128]),
            0.0,
            0,
            2,
            np.array(
                [0, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800], dtype=">u2"
            ),
        ),
        (np.arange(0, 4), 0.0, 0, 0, np.arange(0, 4, dtype=">u2")),
    ],
)
def test_simple_packing_encoding(data, r, e, d, expected):
    encoder = SimplePackingEncoder(r, e, d, 16).data(data)
    actual = encoder.encode()
    np.testing.assert_array_equal(actual, expected)


def test_sect5_writing():
    data = np.arange(0, 16)
    encoder = SimplePackingEncoder(1.0, 1, 1, 16).data(data)

    with BytesIO() as f:
        encoder.write_sect5(f)
        actual = f.getvalue()

    expected = (
        b"\x00\x00\x00\x15\x05\x00\x00\x00\x10\x00\x00\x3f\x80\x00\x00\x00"
        + b"\x01\x00\x01\x10\x01"
    )
    np.testing.assert_array_equal(actual, expected)


def test_sect7_writing():
    data = np.arange(0, 4)
    encoder = SimplePackingEncoder(0.0, 0, 0, 16).data(data)

    with BytesIO() as f:
        encoder.write_sect7(f)
        actual = f.getvalue()

    expected = b"\x00\x00\x00\x0d\x07\x00\x00\x00\x01\x00\x02\x00\x03"
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "data,r,e,d,expectation,error_message",
    [
        (np.arange(0, 4), 0.0, 0, 0, does_not_raise(), None),
        (
            np.array([np.nan, 0]),
            0.0,
            0,
            0,
            pytest.raises(RuntimeError),
            "data contains NaN values",
        ),
    ],
)
def test_errors_in_sect7_writing(data, r, e, d, expectation, error_message):
    encoder = SimplePackingEncoder(r, e, d, 16).data(data)
    with expectation as e:
        with BytesIO() as f:
            encoder.write_sect5(f)

    if e is not None:
        assert str(e.value) == error_message
