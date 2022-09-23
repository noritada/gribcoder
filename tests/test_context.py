import io
from datetime import datetime
from typing import BinaryIO

import helpers
import pytest

from gribgen.context import Grib2MessageWriter
from gribgen.encoders import BaseEncoder
from gribgen.message import Identification, Indicator


class DummyEncoder(BaseEncoder):
    def write_sect5(self, f: BinaryIO):
        pass

    def write_sect6(self, f: BinaryIO):
        pass

    def write_sect7(self, f: BinaryIO):
        pass


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
                encoder = DummyEncoder()
                for num in order:
                    if num == 0:
                        grib2._write_sect1(ind)
                    elif num == 1:
                        grib2._write_sect1(ident)
                    elif num == 2:
                        grib2._write_sect2()
                    elif num == 3:
                        grib2._write_sect3()
                    elif num == 4:
                        grib2._write_sect4()
                    elif num == 5:
                        grib2._write_sect5(encoder)
                    elif num == 6:
                        grib2._write_sect6(encoder)
                    elif num == 7:
                        grib2._write_sect7(encoder)
                    elif num == 8:
                        grib2._write_sect8()

        if e is not None:
            assert str(e.value) == "wrong section order"
