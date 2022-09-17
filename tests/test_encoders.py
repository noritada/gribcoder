import numpy as np

from gribgen.encoders import SimplePackingEncoder


def test_simple_packing_encoding():
    data = np.array([0, 0.25, 0.50, 1, 2, 4, 8, 16, 32, 64, 128])
    encoder = SimplePackingEncoder(0.0, 0, 2, 16)
    actual = encoder.encode(data)
    expected = np.array(
        [0, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800], dtype=">u2"
    )
    np.testing.assert_array_equal(actual, expected)
