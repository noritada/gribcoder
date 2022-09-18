from io import BytesIO

import numpy as np
import pytest

from gribgen.encoders import SimplePackingEncoder


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


def test_sect7_writing():
    data = np.arange(0, 4)
    encoder = SimplePackingEncoder(0.0, 0, 0, 16).data(data)

    with BytesIO() as f:
        encoder.write_sect7(f)
        actual = f.getvalue()

    expected = b"\x00\x00\x00\x0d\x07\x00\x00\x00\x01\x00\x02\x00\x03"
    np.testing.assert_array_equal(actual, expected)
