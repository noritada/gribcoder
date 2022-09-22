import numpy as np
import pytest

from gribgen.grid import LatitudeLongitudeGrid


@pytest.mark.parametrize(
    "input_lat,input_lon,expected_params",
    [
        (
            np.tile(np.arange(10, 20).astype(float), (25, 1)).T,
            np.tile(np.arange(0, 50, 2).astype(float), (10, 1)),
            {
                "first_lat": 10_000_000,
                "first_lon": 0,
                "last_lat": 19_000_000,
                "last_lon": 48_000_000,
                "num_grid_lat": 10,
                "num_grid_lon": 25,
                "inc_lat": 1_000_000,
                "inc_lon": 2_000_000,
                "scan_flag": 0b01000000,
            },
        ),
        (
            np.tile(np.arange(20, 10, -1).astype(float), (25, 1)).T,
            np.tile(np.arange(0, 50, 2).astype(float), (10, 1)),
            {
                "first_lat": 20_000_000,
                "first_lon": 0,
                "last_lat": 11_000_000,
                "last_lon": 48_000_000,
                "num_grid_lat": 10,
                "num_grid_lon": 25,
                "inc_lat": 1_000_000,
                "inc_lon": 2_000_000,
                "scan_flag": 0b00000000,
            },
        ),
        (
            np.tile(np.arange(10, 20).astype(float), (25, 1)).T,
            np.tile(np.arange(50, 0, -2).astype(float), (10, 1)),
            {
                "first_lat": 10_000_000,
                "first_lon": 50_000_000,
                "last_lat": 19_000_000,
                "last_lon": 2_000_000,
                "num_grid_lat": 10,
                "num_grid_lon": 25,
                "inc_lat": 1_000_000,
                "inc_lon": 2_000_000,
                "scan_flag": 0b11000000,
            },
        ),
        (
            np.tile(np.arange(20, 10, -1).astype(float), (25, 1)).T,
            np.tile(np.arange(50, 0, -2).astype(float), (10, 1)),
            {
                "first_lat": 20_000_000,
                "first_lon": 50_000_000,
                "last_lat": 11_000_000,
                "last_lon": 2_000_000,
                "num_grid_lat": 10,
                "num_grid_lon": 25,
                "inc_lat": 1_000_000,
                "inc_lon": 2_000_000,
                "scan_flag": 0b10000000,
            },
        ),
        (
            np.tile(np.arange(10, 20).astype(float), (25, 1)),
            np.tile(np.arange(0, 50, 2).astype(float), (10, 1)).T,
            {
                "first_lat": 10_000_000,
                "first_lon": 0,
                "last_lat": 19_000_000,
                "last_lon": 48_000_000,
                "num_grid_lat": 10,
                "num_grid_lon": 25,
                "inc_lat": 1_000_000,
                "inc_lon": 2_000_000,
                "scan_flag": 0b01100000,
            },
        ),
        (
            np.tile(np.arange(20, 10, -1).astype(float), (25, 1)),
            np.tile(np.arange(0, 50, 2).astype(float), (10, 1)).T,
            {
                "first_lat": 20_000_000,
                "first_lon": 0,
                "last_lat": 11_000_000,
                "last_lon": 48_000_000,
                "num_grid_lat": 10,
                "num_grid_lon": 25,
                "inc_lat": 1_000_000,
                "inc_lon": 2_000_000,
                "scan_flag": 0b00100000,
            },
        ),
        (
            np.tile(np.arange(10, 20).astype(float), (25, 1)),
            np.tile(np.arange(50, 0, -2).astype(float), (10, 1)).T,
            {
                "first_lat": 10_000_000,
                "first_lon": 50_000_000,
                "last_lat": 19_000_000,
                "last_lon": 2_000_000,
                "num_grid_lat": 10,
                "num_grid_lon": 25,
                "inc_lat": 1_000_000,
                "inc_lon": 2_000_000,
                "scan_flag": 0b11100000,
            },
        ),
        (
            np.tile(np.arange(20, 10, -1).astype(float), (25, 1)),
            np.tile(np.arange(50, 0, -2).astype(float), (10, 1)).T,
            {
                "first_lat": 20_000_000,
                "first_lon": 50_000_000,
                "last_lat": 11_000_000,
                "last_lon": 2_000_000,
                "num_grid_lat": 10,
                "num_grid_lon": 25,
                "inc_lat": 1_000_000,
                "inc_lon": 2_000_000,
                "scan_flag": 0b10100000,
            },
        ),
    ],
)
def test_lat_lon_grid_parameter_extraction_from_ndarrays(
    input_lat, input_lon, expected_params
):
    actual = LatitudeLongitudeGrid.from_ndarrays(input_lat, input_lon)
    expected = LatitudeLongitudeGrid(**expected_params)
    assert actual == expected
