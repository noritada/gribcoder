import numpy as np
import pytest

from gribgen.utils import bytes_from_float, bytes_from_int, create_sect_header, grib_int


def test_sect_header_creation():
    actual = create_sect_header(5, 255).tobytes()
    expected = b"\x00\x00\x00\xff\x05"
    assert actual == expected


@pytest.mark.parametrize(
    "input_,byte_length,expected",
    [
        (1024, 4, 0x00000400),
        (-1024, 4, 0x80000400),
    ],
)
def test_grib_int(input_, byte_length, expected):
    actual = grib_int(input_, byte_length)
    assert actual == expected


def test_bytes_from_positive_int():
    input_ = 1024
    length = 8
    actual = bytes_from_int(input_, length)
    expected = np.array([0, 0, 0, 0, 0, 0, 4, 0])
    np.testing.assert_array_equal(actual, expected)


def test_bytes_from_negative_int():
    input_ = -1024
    length = 8
    actual = bytes_from_int(input_, length)
    expected = np.array([128, 0, 0, 0, 0, 0, 4, 0])
    np.testing.assert_array_equal(actual, expected)


def test_bytes_from_float():
    input_ = 1.0
    length = 4
    actual = bytes_from_float(input_, length)
    expected = np.array([0x3F, 0x80, 0x00, 0x00])
    np.testing.assert_array_equal(actual, expected)
