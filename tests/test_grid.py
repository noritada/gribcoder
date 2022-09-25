from io import BytesIO

import helpers
import numpy as np
import pytest

from gribgen.grid import DTYPE_SHAPE_OF_THE_EARTH, LatitudeLongitudeGrid


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


@pytest.mark.parametrize(
    "input,expectation,error_message",
    [
        (
            np.array([(0, 0, 0, 0, 0, 0, 0)], dtype=DTYPE_SHAPE_OF_THE_EARTH),
            helpers.does_not_raise(),
            None,
        ),
        (
            np.array([0]),
            pytest.raises(RuntimeError),
            "wrong dtype",
        ),
        (
            np.array(
                [(0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1)],
                dtype=DTYPE_SHAPE_OF_THE_EARTH,
            ),
            pytest.raises(RuntimeError),
            "wrong length",
        ),
    ],
)
def test_setting_shape_of_the_earth_to_lat_lon_grid(input, expectation, error_message):
    dummy_params = {
        "first_lat": 10_000_000,
        "first_lon": 0,
        "last_lat": 19_000_000,
        "last_lon": 48_000_000,
        "num_grid_lat": 10,
        "num_grid_lon": 25,
        "inc_lat": 1_000_000,
        "inc_lon": 2_000_000,
        "scan_flag": 0b00000000,
    }
    grid = LatitudeLongitudeGrid(**dummy_params)
    with expectation as e:
        grid.shape_of_the_earth(input)

    if e is not None:
        assert str(e.value) == error_message


@pytest.mark.parametrize(
    "input_params,shape_of_the_earth_params,expected",
    [
        (
            # params for JMA MSM GRIB2
            {
                "first_lat": 47_975_000,
                "first_lon": 120_031_250,
                "last_lat": 20_025_000,
                "last_lon": 149_968_750,
                "num_grid_lat": 560,
                "num_grid_lon": 480,
                "inc_lat": 50_000,
                "inc_lon": 62_500,
                "scan_flag": 0b00000000,
            },
            np.array(
                [(6, 0xFF, 0xFFFFFFFF, 0xFF, 0xFFFFFFFF, 0xFF, 0xFFFFFFFF)],
                dtype=DTYPE_SHAPE_OF_THE_EARTH,
            ),
            b"\x00\x00\x00\x48\x03\x00\x00\x04\x1a\x00\x00\x00\x00\x00\x06\xff"
            + b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00"
            + b"\x01\xe0\x00\x00\x02\x30\x00\x00\x00\x00\xff\xff\xff\xff\x02\xdc"
            + b"\x0a\x58\x07\x27\x88\x12\x30\x01\x31\x8e\xa8\x08\xf0\x57\x6e\x00"
            + b"\x00\xf4\x24\x00\x00\xc3\x50\x00",
        ),
    ],
)
def test_sect7_writing(input_params, shape_of_the_earth_params, expected):
    grid = LatitudeLongitudeGrid(**input_params).shape_of_the_earth(
        shape_of_the_earth_params
    )
    with BytesIO() as f:
        grid.write(f)
        actual = f.getvalue()
    assert actual == expected
