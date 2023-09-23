from io import BytesIO

import helpers
import numpy as np
import pytest

from gribcoder import SimplePackingEncoder
from gribcoder.encoders import create_bitmap


@pytest.mark.parametrize(
    "input,expected_r,expected_e,expected_d,expected_n,expected_encoded,"
    "expected_restored",
    [
        (
            np.arange(256),
            0.0,
            0,
            0,
            8,
            np.arange(256),
            np.arange(256),
        ),
        (
            np.arange(256) + 100,
            100.0,
            0,
            0,
            8,
            np.arange(256),
            np.arange(256) + 100,
        ),
        (
            np.arange(256) * -1,
            -255.0,
            0,
            0,
            8,
            np.arange(256)[::-1],
            np.arange(256) * -1,
        ),
        (
            np.arange(256) * 100,
            0.0,
            0,
            -2,
            8,
            np.arange(256),
            np.arange(256) * 100,
        ),
        (
            (np.arange(256) + 100) / 100,
            100.0,
            0,
            2,
            8,
            np.arange(256),
            (np.arange(256) + 100) / 100,
        ),
        (
            (np.arange(256) - 100) / 100,
            -100.0,
            0,
            2,
            8,
            np.arange(256),
            (np.arange(256) - 100) / 100,
        ),
        (
            np.array([0, 1, 382, 383]),
            0.0,
            0,
            -1,
            8,
            np.array([0, 0, 38, 38]),
            np.array([0, 0, 380, 380]),
        ),
        (
            np.ma.MaskedArray(np.arange(256), mask=[0] + [1] * 254 + [0]),
            0.0,
            0,
            0,
            8,
            np.array([0, 255]),
            np.array([0, 255]),
        ),
        (
            np.ma.MaskedArray(np.arange(256), mask=[1] * 256),
            0.0,
            0,
            0,
            0,
            np.array([]),
            np.array([]),
        ),
    ],
)
def test_auto_parametrization_simple_linear(
    input,
    expected_r,
    expected_e,
    expected_d,
    expected_n,
    expected_encoded,
    expected_restored,
):
    encoder = SimplePackingEncoder.auto_parametrized_from(
        input, scaling="simple-linear", nbit=8
    )
    actual_encoded, _actual_bitmap = encoder.encode()
    assert encoder.r == expected_r
    assert encoder.e == expected_e
    assert encoder.d == expected_d
    assert encoder.n == expected_n
    np.testing.assert_array_equal(actual_encoded, expected_encoded)

    actual_restored = decode_simple_packing(
        actual_encoded, encoder.r, encoder.e, encoder.d, encoder.n
    )
    np.testing.assert_array_almost_equal(actual_restored, expected_restored, decimal=15)


@pytest.mark.parametrize(
    "input,decimals,expected_r,expected_e,expected_d,expected_n,expected_encoded,"
    "expected_restored",
    [
        (
            np.arange(256),
            0,
            0.0,
            0,
            0,
            8,
            np.arange(256),
            np.arange(256),
        ),
        (
            np.arange(256),
            1,
            0.0,
            0,
            1,
            16,
            np.arange(256) * 10,
            np.arange(256),
        ),
        (
            np.arange(256),
            3,
            0.0,
            0,
            3,
            32,
            np.arange(256) * 1000,
            np.arange(256),
        ),
        (
            np.arange(256),
            8,
            0.0,
            0,
            8,
            64,
            np.arange(256) * 100_000_000,
            np.arange(256),
        ),
        (
            np.arange(256),
            -1,
            0.0,
            0,
            -1,
            8,
            np.round(np.arange(256) * 0.1),
            np.round(np.arange(256), decimals=-1),
            # np.arange(256),
        ),
        (
            np.ma.MaskedArray(np.arange(256), mask=[0] + [1] * 254 + [0]),
            0,
            0.0,
            0,
            0,
            8,
            np.array([0, 255]),
            np.array([0, 255]),
        ),
    ],
)
def test_auto_parametrization_fixed_digit_linear(
    input,
    decimals,
    expected_r,
    expected_e,
    expected_d,
    expected_n,
    expected_encoded,
    expected_restored,
):
    encoder = SimplePackingEncoder.auto_parametrized_from(
        input, scaling="fixed-digit-linear", decimals=decimals
    )
    actual_encoded, _actual_bitmap = encoder.encode()
    assert encoder.r == expected_r
    assert encoder.e == expected_e
    assert encoder.d == expected_d
    assert encoder.n == expected_n
    np.testing.assert_array_equal(actual_encoded, expected_encoded)

    actual_restored = decode_simple_packing(
        actual_encoded, encoder.r, encoder.e, encoder.d, encoder.n
    )
    np.testing.assert_array_almost_equal(actual_restored, expected_restored, decimal=15)


def decode_simple_packing(data, r, e, d, n):
    if n == 0:
        return np.array([])
    else:
        return (r + data.astype(float) * 2**e) * 10 ** (-d)


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
            np.arange(0, 16).reshape(4, 4),
            1.0,
            1,
            1,
            b"\x00\x00\x00\x15\x05\x00\x00\x00\x10\x00\x00\x3f\x80\x00\x00\x00"
            + b"\x01\x00\x01\x10\x01",
        ),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128, 256],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            ),
            0.0,
            0,
            2,
            b"\x00\x00\x00\x15\x05\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00"
            + b"\x00\x00\x02\x10\x00",
        ),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128, 256],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            ).reshape(4, 3),
            0.0,
            0,
            2,
            b"\x00\x00\x00\x15\x05\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00"
            + b"\x00\x00\x02\x10\x00",
        ),
    ],
)
def test_sect5_writing(input, r, e, d, expected):
    encoder = SimplePackingEncoder(r, e, d, 16).input(input)
    with BytesIO() as f:
        encoder.write_sect5(f)
        actual = f.getvalue()
    assert actual == expected


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
            np.arange(0, 4).reshape(2, 2),
            0.0,
            0,
            0,
            b"\x00\x00\x00\x06\x06\xff",
        ),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128, 256],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            ),
            0.0,
            0,
            2,
            b"\x00\x00\x00\x08\x06\x00\xb3\x30",
        ),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128, 256],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            ).reshape(4, 3),
            0.0,
            0,
            2,
            b"\x00\x00\x00\x08\x06\x00\xb3\x30",
        ),
    ],
)
def test_sect6_writing(input, r, e, d, expected):
    encoder = SimplePackingEncoder(r, e, d, 16).input(input)
    with BytesIO() as f:
        encoder.write_sect6(f)
        actual = f.getvalue()
    assert actual == expected


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
            np.arange(0, 4).reshape(2, 2),
            0.0,
            0,
            0,
            b"\x00\x00\x00\x0d\x07\x00\x00\x00\x01\x00\x02\x00\x03",
        ),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128, 256],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            ),
            0.0,
            0,
            2,
            b"\x00\x00\x00\x13\x07\x00\x00\x00\x32\x00\x64\x03\x20\x06\x40\x32"
            + b"\x00\x64\x00",
        ),
        (
            np.ma.array(
                [0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128, 256],
                mask=[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            ).reshape(4, 3),
            0.0,
            0,
            2,
            b"\x00\x00\x00\x13\x07\x00\x00\x00\x32\x00\x64\x03\x20\x06\x40\x32"
            + b"\x00\x64\x00",
        ),
    ],
)
def test_sect7_writing(input, r, e, d, expected):
    encoder = SimplePackingEncoder(r, e, d, 16).input(input)
    with BytesIO() as f:
        encoder.write_sect7(f)
        actual = f.getvalue()
    assert actual == expected


@pytest.mark.parametrize(
    "input,r,e,d,expectation,error_message",
    [
        (np.arange(0, 4), 0.0, 0, 0, helpers.does_not_raise(), None),
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
