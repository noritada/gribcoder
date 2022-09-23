import dataclasses
from abc import ABC, abstractmethod
from typing import BinaryIO

import numpy as np
from nptyping import Bool, NDArray, Shape, UInt8

from gribgen.utils import SECT_HEADER_DTYPE, create_sect_header, grib_signed


class BaseEncoder(ABC):
    @abstractmethod
    def write_sect5(self, f: BinaryIO) -> int:
        return 0

    @abstractmethod
    def write_sect6(self, f: BinaryIO) -> int:
        return 0

    @abstractmethod
    def write_sect7(self, f: BinaryIO) -> int:
        return 0


@dataclasses.dataclass
class SimplePackingEncoder(BaseEncoder):
    r: float
    e: int
    d: int
    n: int

    def input(self, data: np.ndarray):  # `-> Self` for Python >=3.11 (PEP 673)
        """Sets input data to be encoded.

        The input must be an instance of `np.ndarray` or `np.ma.MaskedArray`. If it is
        an `np.ma.MaskedArray`, a bitmap is also created in the process of the
        encoding; otherwise, no bitmap is created.
        """
        self._input = data
        self._encoded = None
        return self

    def encode(self) -> tuple[np.ndarray, np.ndarray]:
        """Packs and returns two `np.ndarray`s. The first one is an array containing
        encoded values with `n` bits occupied for each element, and the second one is a
        bitmap array."""
        if self._encoded is not None:
            return (self._encoded, self._bitmap)
        if self._input is None:
            raise RuntimeError("data is not specified")
        if isinstance(self._input, np.ma.MaskedArray):
            input_ = self._input[~self._input.mask]
            bitmap = create_bitmap(self._input.mask)
        else:
            input_ = self._input
            bitmap = None
        if np.isnan(input_).any():
            # if the data contains NaN, encoding itself succeeds, but proper values
            # cannot be written out, so we raise an exception
            raise RuntimeError("data contains NaN values")
        dtype = self._determine_dtype()
        encoded = (input_ * 10**self.d - self.r) * 2 ** (-self.e)
        self._encoded = np.round(encoded).astype(dtype)
        self._bitmap = bitmap
        return (self._encoded, self._bitmap)

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

    def write_sect5(self, f: BinaryIO) -> int:
        """Writes parameter data to the stream as Section 5 octet sequence."""
        encoded, _ = self.encode()
        num_of_values = len(encoded)
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
                ("binary_scale_factor", ">u2"),  # grib_signed
                ("decimal_scale_factor", ">u2"),  # grib_signed
                ("bits_per_value", "u1"),
                ("type_of_original_field_values", "u1"),
            ]
        )
        original_data_dtype = self._input.dtype
        if np.issubdtype(original_data_dtype, np.floating):
            field_type = 0
        elif np.issubdtype(original_data_dtype, np.integer):
            field_type = 1
        else:
            raise RuntimeError("unexpected dtype for original data values")

        template_buf = np.array(
            [
                (
                    self.r,
                    grib_signed(self.e, 2),
                    grib_signed(self.d, 2),
                    self.n,
                    field_type,
                )
            ],
            dtype=template_dtype,
        )

        sect_len = (
            SECT_HEADER_DTYPE.itemsize + main_dtype.itemsize + template_dtype.itemsize
        )

        header = create_sect_header(5, sect_len)
        f.write(header)
        f.write(main_buf)
        f.write(template_buf)
        return sect_len

    def write_sect6(self, f: BinaryIO) -> int:
        """Writes bitmap data to the stream as Section 6 octet sequence."""
        _, bitmap = self.encode()

        main_dtype = np.dtype(
            [
                ("bitmap_indicator", "u1"),
            ]
        )
        if bitmap is None:
            main_buf = np.array([(0xFF)], dtype=main_dtype)
            bitmap = np.array([], dtype=">u8")
        else:
            main_buf = np.array([(0x00)], dtype=main_dtype)

        sect_len = SECT_HEADER_DTYPE.itemsize + main_dtype.itemsize + len(bitmap)
        header = create_sect_header(6, sect_len)
        f.write(header)
        f.write(main_buf)
        f.write(bitmap)
        return sect_len

    def write_sect7(self, f: BinaryIO) -> int:
        """Writes encoded data to the stream as Section 7 octet sequence."""
        encoded, _ = self.encode()
        encoded = encoded.tobytes()
        sect_len = SECT_HEADER_DTYPE.itemsize + len(encoded)
        header = create_sect_header(7, sect_len)
        f.write(header)
        f.write(encoded)
        return sect_len


def create_bitmap(mask: NDArray[Shape["*"], Bool]) -> NDArray[Shape["*"], UInt8]:
    """Creates a bitmap octets corresponding to `mask`.

    In the input `mask`, True (1) must means that data is masked (missing).
    In the output bitmap, 0 means data is missing and 1 means data is present."""
    extra_len = len(mask) % 8
    n_pad = 0 if extra_len == 0 else 8 - extra_len
    array = np.pad(mask, (0, n_pad), constant_values=True).reshape(-1, 8)
    bits = np.packbits(~array, axis=-1).ravel()
    return bits
