from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import BinaryIO

import numpy as np

from .utils import SECT_HEADER_DTYPE, create_sect_header, grib_signed


class BaseGrid(ABC):
    @abstractmethod
    def write(self, f: BinaryIO) -> int:
        return 0


DTYPE_SHAPE_OF_THE_EARTH = np.dtype(
    [
        ("shape_of_the_earth", "u1"),
        ("scale_factor_of_radius_of_spherical_earth", "u1"),
        ("scaled_value_of_radius_of_spherical_earth", ">u4"),
        ("scale_factor_of_earth_major_axis", "u1"),
        ("scaled_value_of_earth_major_axis", ">u4"),
        ("scale_factor_of_earth_minor_axis", "u1"),
        ("scaled_value_of_earth_minor_axis", ">u4"),
    ]
)

DTYPE_SECTION_3 = np.dtype(
    [
        ("source_of_grid_definition", "u1"),
        ("number_of_data_points", ">u4"),
        ("number_of_octets_for_number_of_points", "u1"),
        ("interpretation_of_number_of_points", "u1"),
        ("grid_definition_template_number", ">u2"),
    ]
)

DTYPE_TEMPLATE_3_0_MAIN = np.dtype(
    [
        ("n_i", ">u4"),
        ("n_j", ">u4"),
        ("basic_angle_of_the_initial_production_domain", ">u4"),
        ("subdivisions_of_basic_angle", ">u4"),
        ("latitude_of_first_grid_point", ">u4"),  # grib_signed
        ("longitude_of_first_grid_point", ">u4"),  # grib_signed
        ("resolution_and_component_flags", "u1"),
        ("latitude_of_last_grid_point", ">u4"),  # grib_signed
        ("longitude_of_last_grid_point", ">u4"),  # grib_signed
        ("i_direction_increment", ">u4"),
        ("j_direction_increment", ">u4"),
        ("scanning_mode", "u1"),
    ]
)

_UNIT_DEG = 1_000_000


@dataclasses.dataclass
class LatitudeLongitudeGrid(BaseGrid):
    first_lat: int
    first_lon: int
    last_lat: int
    last_lon: int
    num_grid_lat: int
    num_grid_lon: int
    inc_lat: int | None
    inc_lon: int | None
    scan_flag: int

    @classmethod
    def from_ndarrays(cls, lat: np.ndarray, lon: np.ndarray):
        if lat.ndim == 2 and lon.ndim == 2:
            scan_flag = 0b00000000

            if (
                np.unique(lat, axis=1).shape[1] == 1
                and np.unique(lon, axis=0).shape[0] == 1
            ):
                # consecutive in direction i
                lat_1d = lat[:, 0]
                lon_1d = lon[0, :]
            elif (
                np.unique(lat, axis=0).shape[0] == 1
                and np.unique(lon, axis=1).shape[1] == 1
            ):
                # consecutive in direction j
                lat_1d = lat[0, :]
                lon_1d = lon[:, 0]
                scan_flag = scan_flag | 0b00100000
            else:
                raise RuntimeError("scanning direction switching is not supported")

            corners = np.array([lat[0, 0], lon[0, 0], lat[-1, -1], lon[-1, -1]])
            corners = np.around(corners * _UNIT_DEG).astype(int)

            num_grid_lat = lat_1d.shape[0]
            num_grid_lon = lon_1d.shape[0]
            lat_calculated = np.linspace(corners[0], corners[2], num_grid_lat)
            lon_calculated = np.linspace(corners[1], corners[3], num_grid_lon)
            if np.allclose(
                lat_calculated, np.around(lat_1d * _UNIT_DEG).astype(int), atol=1
            ) and np.allclose(
                lon_calculated, np.around(lon_1d * _UNIT_DEG).astype(int), atol=1
            ):
                pass
            else:
                raise RuntimeError("latitude or longitude is not evenly spaced")

            inc_lat = lat_calculated[1] - lat_calculated[0]
            inc_lon = lon_calculated[1] - lon_calculated[0]
            if inc_lat < 0:
                inc_lat = -inc_lat
            else:
                scan_flag = scan_flag | 0b01000000
            if inc_lon < 0:
                inc_lon = -inc_lon
                scan_flag = scan_flag | 0b10000000

            return cls(
                corners[0],
                corners[1],
                corners[2],
                corners[3],
                num_grid_lat,
                num_grid_lon,
                inc_lat,
                inc_lon,
                scan_flag,
            )
        else:
            raise RuntimeError(
                "construction from ndarrays othar than 2d arrays is not supported"
            )

    def shape_of_the_earth(
        self, values: np.ndarray
    ):  # `-> Self` for Python >=3.11 (PEP 673)
        if values.dtype != DTYPE_SHAPE_OF_THE_EARTH:
            raise RuntimeError("wrong dtype")
        if len(values) != 1:
            raise RuntimeError("wrong length")
        self._shape_of_the_earth = values
        return self

    def write(self, f: BinaryIO) -> int:
        section_main_buf = np.array(
            [
                (
                    0,  # using lat/lon grid in code table 3.1
                    self.num_grid_lat * self.num_grid_lon,
                    0,  # fixed for regular grids
                    0,  # fixed for regular grids
                    0,  # lat/lon grid
                )
            ],
            dtype=DTYPE_SECTION_3,
        )

        template_main_buf = np.array(
            [
                (
                    self.num_grid_lon,
                    self.num_grid_lat,
                    0,
                    0xFFFFFFFF,
                    grib_signed(self.first_lat, 4),
                    grib_signed(self.first_lon, 4),
                    self._get_resolution_and_component_flag(),
                    grib_signed(self.last_lat, 4),
                    grib_signed(self.last_lon, 4),
                    self.inc_lon,
                    self.inc_lat,
                    self.scan_flag,
                )
            ],
            dtype=DTYPE_TEMPLATE_3_0_MAIN,
        )

        sect_len = (
            SECT_HEADER_DTYPE.itemsize
            + DTYPE_SECTION_3.itemsize
            + DTYPE_SHAPE_OF_THE_EARTH.itemsize
            + DTYPE_TEMPLATE_3_0_MAIN.itemsize
        )

        header = create_sect_header(3, sect_len)
        f.write(header)
        f.write(section_main_buf)
        f.write(self._shape_of_the_earth)
        f.write(template_main_buf)
        return sect_len

    def _get_resolution_and_component_flag(self) -> int:
        flag = 0b00000000
        if self.inc_lat is not None:
            flag |= 0b00010000
        if self.inc_lon is not None:
            flag |= 0b00100000
        return flag
