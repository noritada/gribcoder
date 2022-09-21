import pytest

from gribgen.utils import create_sect_header, grib_signed


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
def test_grib_signed(input_, byte_length, expected):
    actual = grib_signed(input_, byte_length)
    assert actual == expected
