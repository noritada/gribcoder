import dataclasses
from datetime import datetime
from typing import BinaryIO

import numpy as np

from .utils import SECT_HEADER_DTYPE, create_sect_header

DTYPE_SECTION_0 = np.dtype(
    [
        ("identifier", ">u4"),  # treat as a number for simplicity
        ("reserved", ">u2"),
        ("discipline", "u1"),
        ("edition_number", "u1"),
        ("total_length", ">u8"),
    ]
)

DTYPE_SECTION_1 = np.dtype(
    [
        ("centre", ">u2"),
        ("sub_centre", ">u2"),
        ("tables_version", "u1"),
        ("local_tables_version", "u1"),
        ("significance_of_reference_time", "u1"),
        ("year", ">u2"),
        ("month", "u1"),
        ("day", "u1"),
        ("hour", "u1"),
        ("minute", "u1"),
        ("second", "u1"),
        ("production_status_of_processed_data", "u1"),
        ("type_of_processed_data", "u1"),
    ]
)


@dataclasses.dataclass
class Indicator:
    discipline: int
    total_length: int = 0

    def write(self, f: BinaryIO) -> int:
        section_buf = np.array(
            [(0x47524942, 0xFFFF, self.discipline, 2, self.total_length)],
            dtype=DTYPE_SECTION_0,
        )

        f.write(section_buf)
        return DTYPE_SECTION_0.itemsize


@dataclasses.dataclass
class Identification:
    centre: int
    sub_centre: int
    tables_version: int
    local_tables_version: int
    significance_of_reftime: int
    reftime: datetime
    production_status_of_processed_data: int
    type_of_processed_data: int

    def write(self, f: BinaryIO) -> int:
        section_buf = np.array(
            [
                (
                    self.centre,
                    self.sub_centre,
                    self.tables_version,
                    self.local_tables_version,
                    self.significance_of_reftime,
                    self.reftime.year,
                    self.reftime.month,
                    self.reftime.day,
                    self.reftime.hour,
                    self.reftime.minute,
                    self.reftime.second,
                    self.production_status_of_processed_data,
                    self.type_of_processed_data,
                )
            ],
            dtype=DTYPE_SECTION_1,
        )

        sect_len = SECT_HEADER_DTYPE.itemsize + DTYPE_SECTION_1.itemsize

        header = create_sect_header(1, sect_len)
        f.write(header)
        f.write(section_buf)
        return sect_len
