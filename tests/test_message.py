from datetime import datetime
from io import BytesIO

from gribcoder import Identification, Indicator


def test_indicator():
    ind = Indicator(0)
    with BytesIO() as f:
        ind.write(f)
        actual = f.getvalue()
    expected = b"GRIB\xff\xff\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
    assert actual == expected


def test_identification():
    ident = Identification(34, 0, 5, 1, 1, datetime(2019, 3, 4, 0, 0, 0), 0, 1)
    with BytesIO() as f:
        ident.write(f)
        actual = f.getvalue()
    expected = (
        b"\x00\x00\x00\x15\x01\x00\x22\x00\x00\x05\x01\x01\x07\xe3\x03\x04"
        + b"\x00\x00\x00\x00\x01"
    )
    assert actual == expected
