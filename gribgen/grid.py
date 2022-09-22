import dataclasses

import numpy as np

_UNIT_DEG = 1_000_000


@dataclasses.dataclass
class LatitudeLongitudeGrid:
    first_lat: int
    first_lon: int
    last_lat: int
    last_lon: int
    num_grid_lat: int
    num_grid_lon: int
    inc_lat: int
    inc_lon: int
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
