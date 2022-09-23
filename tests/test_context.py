import io
from datetime import datetime
from typing import BinaryIO

import helpers
import numpy as np
import pytest

from gribgen.context import Grib2MessageWriter
from gribgen.encoders import BaseEncoder
from gribgen.message import DTYPE_SECTION_0, Identification, Indicator


def fake_write_sect(grib2, sect_num, ind, ident, encoder):
    if sect_num == 0:
        grib2._write_sect1(ind)
    elif sect_num == 1:
        grib2._write_sect1(ident)
    elif sect_num == 2:
        grib2._write_sect2()
    elif sect_num == 3:
        grib2._write_sect3()
    elif sect_num == 4:
        grib2._write_sect4()
    elif sect_num == 5:
        grib2._write_sect5(encoder)
    elif sect_num == 6:
        grib2._write_sect6(encoder)
    elif sect_num == 7:
        grib2._write_sect7(encoder)
    elif sect_num == 8:
        grib2._write_sect8()


class EmptyEncoder(BaseEncoder):
    def write_sect5(self, f: BinaryIO) -> int:
        return 0

    def write_sect6(self, f: BinaryIO) -> int:
        return 0

    def write_sect7(self, f: BinaryIO) -> int:
        return 0


@pytest.mark.parametrize(
    "order,expectation",
    [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8], pytest.raises(RuntimeError)),
        ([0, 1, 2, 3, 4, 5, 6, 7], pytest.raises(RuntimeError)),
        ([1, 2, 3, 4, 5, 6, 7], pytest.raises(RuntimeError)),
        ([2, 3, 4, 5, 6, 7], helpers.does_not_raise()),
        ([3, 4, 5, 6, 7], helpers.does_not_raise()),
        ([4, 5, 6, 7], pytest.raises(RuntimeError)),
        ([5, 6, 7], pytest.raises(RuntimeError)),
        ([6, 7], pytest.raises(RuntimeError)),
        ([7], pytest.raises(RuntimeError)),
        ([2, 3, 4, 5, 6], pytest.raises(RuntimeError)),
        ([3, 4, 5, 6], pytest.raises(RuntimeError)),
        ([4, 5, 6], pytest.raises(RuntimeError)),
        ([5, 6], pytest.raises(RuntimeError)),
        ([6], pytest.raises(RuntimeError)),
        ([3, 4, 5, 6], pytest.raises(RuntimeError)),
        ([4, 5, 6], pytest.raises(RuntimeError)),
        ([5, 6], pytest.raises(RuntimeError)),
        ([6], pytest.raises(RuntimeError)),
        ([2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7], helpers.does_not_raise()),
        ([3, 4, 5, 6, 7, 3, 4, 5, 6, 7], helpers.does_not_raise()),
        ([3, 4, 5, 6, 7, 4, 5, 6, 7], helpers.does_not_raise()),
    ],
)
def test_section_order(order, expectation):
    with io.BytesIO() as fw:
        with expectation as e:
            ind = Indicator(0)
            ident = Identification(0, 0, 0, 0, 0, datetime.now(), 0, 0)
            with Grib2MessageWriter(fw, ind, ident) as grib2:
                encoder = EmptyEncoder()
                for num in order:
                    fake_write_sect(grib2, num, ind, ident, encoder)

        if e is not None:
            assert str(e.value) == "wrong section order"


class ParityBitSizedEncoder(BaseEncoder):
    def write_sect5(self, f: BinaryIO) -> int:
        return self._write_length(f, 0b00001000)

    def write_sect6(self, f: BinaryIO) -> int:
        return self._write_length(f, 0b00010000)

    def write_sect7(self, f: BinaryIO) -> int:
        return self._write_length(f, 0b00100000)

    def _write_length(self, f: BinaryIO, len: int) -> int:
        return f.write(b"\xff" * len)


@pytest.mark.parametrize(
    "order,expected",
    [
        ([2, 3, 4, 5, 6, 7], 16 + 21 + 0 + 0 + 0 + 0b00111000 + 4),
        ([3, 4, 5, 6, 7], 16 + 21 + 0 + 0 + 0b00111000 + 4),
        (
            [2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7],
            16 + 21 + (0 + 0 + 0 + 0b00111000) * 2 + 4,
        ),
        ([3, 4, 5, 6, 7, 3, 4, 5, 6, 7], 16 + 21 + (0 + 0 + 0b00111000) * 2 + 4),
        ([3, 4, 5, 6, 7, 4, 5, 6, 7], 16 + 21 + 0 + (0 + 0b00111000) * 2 + 4),
    ],
)
def test_output_message_length(order, expected):
    with io.BytesIO() as fw:
        ind = Indicator(0)
        ident = Identification(0, 0, 0, 0, 0, datetime.now(), 0, 0)
        with Grib2MessageWriter(fw, ind, ident) as grib2:
            encoder = ParityBitSizedEncoder()
            for num in order:
                fake_write_sect(grib2, num, ind, ident, encoder)

        output = fw.getvalue()

    actual_message_length = len(output)
    assert actual_message_length == expected

    output_sect0 = np.frombuffer(output, dtype=DTYPE_SECTION_0, count=1)
    actual_length_recorded_in_sect0 = output_sect0[0]["total_length"]
    assert actual_length_recorded_in_sect0 == expected
