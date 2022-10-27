import dataclasses
from datetime import datetime
from io import BytesIO

import numpy as np

import gribgen as gribcoder


@dataclasses.dataclass
class Element:
    name: str
    parameter_category: int
    parameter_number: int


ELEMENTS = [
    Element("REFD", 15, 1),
]


def generate_grib2(grid_lats, grid_lons, grid_alts, regridded, time):
    with BytesIO() as fw:
        ind = gribcoder.Indicator(0)
        ident = gribcoder.Identification(0xFFFF, 0xFFFF, 29, 0, 0, time, 0, 0)
        with gribcoder.Grib2MessageWriter(fw, ind, ident) as grib2:
            grid = gribcoder.LatitudeLongitudeGrid.from_ndarrays(
                grid_lats[:, :, 0], grid_lons[:, :, 0]
            ).shape_of_the_earth(
                np.array(
                    [(6, 0xFF, 0xFFFFFFFF, 0xFF, 0xFFFFFFFF, 0xFF, 0xFFFFFFFF)],
                    dtype=gribcoder.DTYPE_SHAPE_OF_THE_EARTH,
                )
            )
            grib2._write_sect3(grid)

            n_alt = grid_alts.shape[2]
            for i in range(n_alt):
                alt = grid_alts[0, 0, i]
                for elem_index, elem in enumerate(ELEMENTS):
                    if elem is None:
                        continue
                    product = (
                        gribcoder.ProductDefinitionWithTemplate4_0(0)
                        .parameter(
                            np.array(
                                [(elem.parameter_category, elem.parameter_number)],
                                dtype=gribcoder.DTYPE_SECTION_4_PARAMETER,
                            )
                        )
                        .generating_process(
                            np.array(
                                [(0, 0xFF, 0xFF)],
                                dtype=gribcoder.DTYPE_SECTION_4_GENERATING_PROCESS,
                            )
                        )
                        .forecast_time(
                            np.array(
                                [(0, 0, 0, 0)],
                                dtype=gribcoder.DTYPE_SECTION_4_FORECAST_TIME,
                            )
                        )
                        .horizontal(
                            np.array(
                                [(102, 0, alt, 0xFF, 0xFF, 0xFFFFFFFF)],
                                dtype=gribcoder.DTYPE_SECTION_4_HORIZONTAL,
                            )
                        )
                    )
                    grib2._write_sect4(product)

                    print(regridded.shape)
                    encoder = gribcoder.SimplePackingEncoder(0, 0, 0, 16).input(
                        regridded[:, :, i, elem_index]
                    )
                    grib2._write_sect5(encoder)
                    grib2._write_sect6(encoder)
                    grib2._write_sect7(encoder)

        output = fw.getvalue()
    return output


grid_lats = np.tile(np.arange(5), (2, 1)).reshape(2, -1, 1)
grid_lons = np.tile(np.arange(2), (5, 1)).T.reshape(-1, 5, 1)
grid_alts = np.ones_like(grid_lats)
x, y, z = grid_lats.shape
length = x * y * z
values = np.arange(length).reshape(grid_lats.shape)

print(grid_lats[:, :, 0])
print(grid_lons[:, :, 0])
print(values[:, :, 0])


grib2 = generate_grib2(
    grid_lats, grid_lons, grid_alts, values.reshape(2, 5, 1, 1), datetime.now()
)
with open("generated.grib", "wb") as fw:
    fw.write(grib2)
