from __future__ import annotations

import contextlib
import dataclasses
from typing import BinaryIO, Callable

from .encoders import BaseEncoder
from .grid import BaseGrid
from .message import Identification, Indicator
from .product import BaseProductDefinition


@dataclasses.dataclass
class Grib2MessageWriter:
    f: BinaryIO
    ind: Indicator
    ident: Identification
    _size: int = dataclasses.field(default=0, init=False)
    _last_sect_no: int = dataclasses.field(default=0, init=False)
    _start_pos: int = dataclasses.field(init=False)

    def __enter__(self):
        self._check_file()
        self._write_sect0()
        self._write_sect1()
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        if exc_type is None:
            self.close()

    def close(self):
        self._write_sect8()
        self._finalize_size()

    def _check_file(self):
        if not self.f.writable():
            raise RuntimeError("file is not writable")
        if not self.f.seekable():
            raise RuntimeError("file is not seekable")
        self._start_pos = self.f.tell()

    def _write_sect0(self):
        with self._section_context(0):
            self._size += self.ind.write(self.f)

    def _write_sect1(self):
        with self._section_context(1):
            self._size += self.ident.write(self.f)

    def _write_sect2(self):
        with self._section_context(2, lambda x: x == 7):
            pass  # not implemented

    def _write_sect3(self, grid: BaseGrid):
        with self._section_context(3, lambda x: x == 1 or x == 7):
            self._size += grid.write(self.f)

    def _write_sect4(self, product: BaseProductDefinition):
        with self._section_context(4, lambda x: x == 7):
            self._size += product.write(self.f)

    def _write_sect5(self, encoder: BaseEncoder):
        with self._section_context(5):
            self._size += encoder.write_sect5(self.f)

    def _write_sect6(self, encoder: BaseEncoder):
        with self._section_context(6):
            self._size += encoder.write_sect6(self.f)

    def _write_sect7(self, encoder: BaseEncoder):
        with self._section_context(7):
            self._size += encoder.write_sect7(self.f)

    def _write_sect8(self):
        with self._section_context(8):
            self._size += self.f.write(b"\x37\x37\x37\x37")

    def _finalize_size(self):
        self.ind.total_length = self._size
        self.f.seek(self._start_pos)
        self.ind.write(self.f)

    @contextlib.contextmanager
    def _section_context(self, sect_no: int, cond: Callable[[int], bool] | None = None):
        expected_prev_sect_no = 0 if sect_no == 0 else sect_no - 1
        if self._last_sect_no != 8 and (
            self._last_sect_no == expected_prev_sect_no
            or (cond is not None and cond(self._last_sect_no))
        ):
            pass
        else:
            raise RuntimeError(
                f"wrong section order: {self._last_sect_no} -> {sect_no}"
            )
        yield
        self._last_sect_no = sect_no
