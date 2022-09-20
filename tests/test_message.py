from datetime import datetime
from io import BytesIO

from gribgen.message import Identification


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
