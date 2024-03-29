from io import BytesIO

import pytest

from gribcoder import (
    NULL_FIXED_SURFACE,
    FixedSurface,
    ProductDefinitionWithTemplate4_0,
    ProductParameter,
)
from gribcoder.utils import write


def test_writing_parameter():
    product = ProductDefinitionWithTemplate4_0(0).parameter(ProductParameter(0, 1))

    with BytesIO() as f:
        write(f, product._parameter)
        actual = f.getvalue()
    expected = b"\x00\x01"

    assert actual == expected


@pytest.mark.parametrize(
    "scale,expected_byte",
    [(0, b"\x00"), (-3, b"\x83")],
)
def test_writing_horizontal(scale, expected_byte):
    product = ProductDefinitionWithTemplate4_0(0).horizontal(
        (FixedSurface(101, scale, 2), NULL_FIXED_SURFACE)
    )

    with BytesIO() as f:
        write(f, product._horizontal)
        actual = f.getvalue()
    expected = b"\x65%b\x00\x00\x00\x02\xff\xff\xff\xff\xff\xff" % (expected_byte)

    assert actual == expected
