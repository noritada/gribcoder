import dataclasses
from typing import BinaryIO

import numpy as np

from gribgen.utils import SECT_HEADER_DTYPE, create_sect_header, grib_int


@dataclasses.dataclass
class SimplePackingEncoder:
    r: float
    e: int
    d: int
    n: int

    def data(self, data: np.ndarray):  # `-> Self` for Python >=3.11 (PEP 673)
        """Sets data to be encoded."""
        self._data = data
        self._encoded = None
        return self

    def encode(self):
        """Packs and return as an `np.ndarray` with `n` bits occupied by each element."""
        if self._encoded is not None:
            return self._encoded
        if self._data is None:
            raise RuntimeError("data is not specified")
        if np.isnan(self._data).any():
            # if the data contains NaN, encoding itself succeeds, but proper values
            # cannot be written out, so we raise an exception
            raise RuntimeError("data contains NaN values")
        dtype = self._determine_dtype()
        encoded = (self._data * 10**self.d - self.r) * 2 ** (-self.e)
        self._encoded = np.round(encoded).astype(dtype)
        return self._encoded

    def _determine_dtype(self):
        if self.n == 8:
            return ">u1"
        elif self.n == 16:
            return ">u2"
        elif self.n == 32:
            return ">u4"
        elif self.n == 64:
            return ">u8"
        else:
            raise RuntimeError("n other than 8, 16, 32, and 64 is not supported")

    def write_sect5(self, f: BinaryIO):
        """Writes parameter data to the stream as Section 5 octet sequence."""
        num_of_values = len(self.encode())
        main_dtype = np.dtype(
            [
                ("num_of_values", ">u4"),
                ("template_num", ">u2"),
            ]
        )
        main_buf = np.array([(num_of_values, 0)], dtype=main_dtype)

        template_dtype = np.dtype(
            [
                ("reference_value", ">f4"),
                ("binary_scale_factor", ">u2"),  # grib_int
                ("decimal_scale_factor", ">u2"),  # grib_int
                ("bits_per_value", "u1"),
                ("type_of_original_field_values", "u1"),
            ]
        )
        original_data_dtype = self._data.dtype
        if np.issubdtype(original_data_dtype, np.floating):
            field_type = 0
        elif np.issubdtype(original_data_dtype, np.integer):
            field_type = 1
        else:
            raise RuntimeError("unexpected dtype for original data values")

        template_buf = np.array(
            [(self.r, grib_int(self.e, 2), grib_int(self.d, 2), self.n, field_type)],
            dtype=template_dtype,
        )

        sect_len = (
            SECT_HEADER_DTYPE.itemsize + main_dtype.itemsize + template_dtype.itemsize
        )

        header = create_sect_header(5, sect_len)
        f.write(header)
        f.write(main_buf)
        f.write(template_buf)

    def write_sect7(self, f: BinaryIO):
        """Writes encoded data to the stream as Section 7 octet sequence."""
        encoded = self.encode().tobytes()
        sect_len = SECT_HEADER_DTYPE.itemsize + len(encoded)
        header = create_sect_header(7, sect_len)
        f.write(header)
        f.write(encoded)
