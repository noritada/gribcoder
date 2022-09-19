from contextlib import contextmanager
from io import BytesIO

import numpy as np
import pytest

from gribgen.encoders import SimplePackingEncoder, create_bitmap


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "input,r,e,d,expected_encoded,expected_bitmap",
    [
        (
            np.array([0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128]),
            0.0,
            0,
            2,
            np.array(
                [0, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800], dtype=">u2"
            ),
            None,
        ),
        (np.arange(0, 4), 0.0, 0, 0, np.arange(0, 4, dtype=">u2"), None),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            ),
            0.0,
            0,
            2,
            np.array([0, 50, 100, 800, 1600, 12800], dtype=">u2"),
            np.array([0b10110011, 0b00100000], dtype=">u2"),
        ),
    ],
)
def test_simple_packing_encoding(input, r, e, d, expected_encoded, expected_bitmap):
    encoder = SimplePackingEncoder(r, e, d, 16).input(input)
    actual_encoded, actual_bitmap = encoder.encode()
    np.testing.assert_array_equal(actual_encoded, expected_encoded)
    np.testing.assert_array_equal(actual_bitmap, expected_bitmap)


@pytest.mark.parametrize(
    "input,r,e,d,expected",
    [
        (
            np.arange(0, 16),
            1.0,
            1,
            1,
            b"\x00\x00\x00\x15\x05\x00\x00\x00\x10\x00\x00\x3f\x80\x00\x00\x00"
            + b"\x01\x00\x01\x10\x01",
        ),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            ),
            0.0,
            0,
            2,
            b"\x00\x00\x00\x15\x05\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00"
            + b"\x00\x00\x02\x10\x00",
        ),
    ],
)
def test_sect5_writing(input, r, e, d, expected):
    encoder = SimplePackingEncoder(r, e, d, 16).input(input)
    with BytesIO() as f:
        encoder.write_sect5(f)
        actual = f.getvalue()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "input,r,e,d,expected",
    [
        (
            np.arange(0, 4),
            0.0,
            0,
            0,
            b"\x00\x00\x00\x06\x06\xff",
        ),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            ),
            0.0,
            0,
            2,
            b"\x00\x00\x00\x08\x06\x00\xb3\x20",
        ),
    ],
)
def test_sect6_writing(input, r, e, d, expected):
    encoder = SimplePackingEncoder(r, e, d, 16).input(input)
    with BytesIO() as f:
        encoder.write_sect6(f)
        actual = f.getvalue()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "input,r,e,d,expected",
    [
        (
            np.arange(0, 4),
            0.0,
            0,
            0,
            b"\x00\x00\x00\x0d\x07\x00\x00\x00\x01\x00\x02\x00\x03",
        ),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            ),
            0.0,
            0,
            2,
            b"\x00\x00\x00\x11\x07\x00\x00\x00\x32\x00\x64\x03\x20\x06\x40\x32\x00",
        ),
    ],
)
def test_sect7_writing(input, r, e, d, expected):
    encoder = SimplePackingEncoder(r, e, d, 16).input(input)
    with BytesIO() as f:
        encoder.write_sect7(f)
        actual = f.getvalue()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "input,r,e,d,expectation,error_message",
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
def test_errors_in_sect7_writing(input, r, e, d, expectation, error_message):
    encoder = SimplePackingEncoder(r, e, d, 16).input(input)
    with expectation as e:
        with BytesIO() as f:
            encoder.write_sect5(f)

    if e is not None:
        assert str(e.value) == error_message


@pytest.mark.parametrize(
    "input,expected",
    [
        ([True, False, True, True, False, False], [0b01001100]),
        (
            [
                True,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            [0b01001100, 0b01110000],
        ),
        (
            [
                True,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
            ],
            [0b01001100, 0b01110000, 0b11110000],
        ),
    ],
)
def test_bitmap_creation(input, expected):
    input = np.array(input)
    actual = create_bitmap(input)
    expected = np.array(expected, dtype=np.uint8)
    np.testing.assert_array_equal(actual, expected)
