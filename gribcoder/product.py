import dataclasses
from abc import ABC, abstractmethod
from typing import BinaryIO, NamedTuple

import numpy as np

from .utils import SECT_HEADER_DTYPE, create_sect_header, grib_signed, write

DTYPE_SECTION_4 = np.dtype(
    [
        ("nv", ">u2"),
        ("product_definition_template_number", ">u2"),
    ]
)

DTYPE_SECTION_4_PARAMETER = np.dtype(
    [
        ("parameter_category", "u1"),
        ("parameter_number", "u1"),
    ]
)

DTYPE_SECTION_4_GENERATING_PROCESS = np.dtype(
    [
        ("type_of_generating_process", "u1"),
        ("background_process", "u1"),
        ("generating_process_identifier", "u1"),
    ]
)

DTYPE_SECTION_4_FORECAST_TIME = np.dtype(
    [
        ("hours_after_data_cutoff", ">u2"),
        ("minutes_after_data_cutoff", "u1"),
        ("indicator_of_unit_of_time_range", "u1"),
        ("forecast_time", ">u4"),  # grib_signed
    ]
)

_DTYPE_SECTION_4_FIXED_SURFACE = np.dtype(
    [
        ("type_of_fixed_surface", "u1"),
        ("scale_factor_of_fixed_surface", "u1"),  # grib_signed
        ("scale_value_of_fixed_surface", ">u4"),
    ]
)


class FixedSurface(NamedTuple):
    type: int
    scale_factor: int
    scale_value: int


NULL_FIXED_SURFACE = FixedSurface(0xFF, 0xFF, 0xFFFFFFFF)


class BaseProductDefinition(ABC):
    @abstractmethod
    def write(self, f: BinaryIO) -> int:
        return 0


@dataclasses.dataclass
class ProductDefinitionWithTemplate4_0:
    nv: int

    def parameter(self, values: np.ndarray):  # `-> Self` for Python >=3.11 (PEP 673)
        if values.dtype != DTYPE_SECTION_4_PARAMETER:
            raise RuntimeError("wrong dtype")
        if len(values) != 1:
            raise RuntimeError("wrong length")
        self._parameter = values
        return self

    def generating_process(
        self, values: np.ndarray
    ):  # `-> Self` for Python >=3.11 (PEP 673)
        if values.dtype != DTYPE_SECTION_4_GENERATING_PROCESS:
            raise RuntimeError("wrong dtype")
        if len(values) != 1:
            raise RuntimeError("wrong length")
        self._generating_process = values
        return self

    def forecast_time(
        self, values: np.ndarray
    ):  # `-> Self` for Python >=3.11 (PEP 673)
        if values.dtype != DTYPE_SECTION_4_FORECAST_TIME:
            raise RuntimeError("wrong dtype")
        if len(values) != 1:
            raise RuntimeError("wrong length")
        self._forecast_time = values
        return self

    def horizontal(
        self, surfaces: tuple[FixedSurface | None, FixedSurface | None]
    ):  # `-> Self` for Python >=3.11 (PEP 673)
        surfaces_ = [
            [NULL_FIXED_SURFACE]
            if fs is None
            else [
                FixedSurface(
                    fs.type,
                    grib_signed(fs.scale_factor, 1),
                    fs.scale_value,
                )
            ]
            for fs in surfaces
        ]

        values = np.array(surfaces_, dtype=_DTYPE_SECTION_4_FIXED_SURFACE)

        if len(values) != 2:
            raise RuntimeError("wrong length")
        self._horizontal = values
        return self

    def write(self, f: BinaryIO) -> int:
        section_main_buf = np.array(
            [(self.nv, 0)],
            dtype=DTYPE_SECTION_4,
        )

        sect_len = (
            SECT_HEADER_DTYPE.itemsize
            + DTYPE_SECTION_4.itemsize
            + DTYPE_SECTION_4_PARAMETER.itemsize
            + DTYPE_SECTION_4_GENERATING_PROCESS.itemsize
            + DTYPE_SECTION_4_FORECAST_TIME.itemsize
            + _DTYPE_SECTION_4_FIXED_SURFACE.itemsize * 2
        )

        header = create_sect_header(4, sect_len)
        write(f, header)
        write(f, section_main_buf)
        write(f, self._parameter)
        write(f, self._generating_process)
        write(f, self._forecast_time)
        write(f, self._horizontal)
        return sect_len
