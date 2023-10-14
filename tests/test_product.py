from io import BytesIO

import numpy as np
import pytest

from gribcoder import DTYPE_SECTION_4_FIXED_SURFACE, ProductDefinitionWithTemplate4_0
from gribcoder.utils import grib_signed, write


@pytest.mark.parametrize(
    "scale,expected_byte",
    [(0, b"\x00"), (-3, b"\x83")],
)
def test_writing_horizontal(scale, expected_byte):
    product = ProductDefinitionWithTemplate4_0(0).horizontal(
        np.array(
            [
                [(101, grib_signed(scale, 1), 2)],
                [(0xFF, 0xFF, 0xFFFFFFFF)],
            ],
            dtype=DTYPE_SECTION_4_FIXED_SURFACE,
        )
    )

    with BytesIO() as f:
        write(f, product._horizontal)
        actual = f.getvalue()
    expected = b"\x65%b\x00\x00\x00\x02\xff\xff\xff\xff\xff\xff" % (expected_byte)

    assert actual == expected
