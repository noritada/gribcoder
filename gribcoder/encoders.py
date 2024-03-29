from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from math import ceil, log2
from typing import BinaryIO

import numpy as np
from nptyping import Bool, NDArray, Shape, UInt8

from .utils import SECT_HEADER_DTYPE, create_sect_header, grib_signed, write


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

    @classmethod
    def auto_parametrized_from(
        cls, data: np.ndarray, scaling: str = "simple-linear", **kwargs
    ):
        """Constructs an encoder with parameter sets.

        - "simple-linear" prepares an encoder with parameter sets for linear scaling for
          given "nbit"
        - "fixed-digit-linear" prepares an encoder with parameter sets for linear
          scaling for given "decimals" (number of decimal places; precision)
        """
        if np.ma.isMaskedArray(data) and data.mask.all():
            r, d, n = 0.0, 0, 0
        elif np.ma.isMaskedArray(data) and _is_unique(values := data[~data.mask]):
            r, d, n = values[0], 0, 0
        elif _is_unique(data):
            r, d, n = data[0], 0, 0
        elif scaling == "simple-linear":
            n = kwargs["nbit"]
            r, d = _get_parameters_simple_linear(data, n)
        elif scaling == "fixed-digit-linear":
            d = kwargs["decimals"]
            r, n = _get_parameters_fixed_digit_linear(data, d)
        else:
            raise RuntimeError(f"unsupported scaling type: {scaling}")

        return cls(r, 0, d, n).input(data)

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
            bitmap = create_bitmap(self._input.mask.ravel())
        else:
            input_ = self._input.ravel()
            bitmap = None
        if np.isnan(input_).any():
            # if the data contains NaN, encoding itself succeeds, but proper values
            # cannot be written out, so we raise an exception
            raise RuntimeError("data contains NaN values")
        self._len = len(input_)
        dtype = self._determine_dtype()
        if self.n == 0:
            self._encoded = np.array([], dtype=dtype)
        else:
            encoded = (input_ * 10**self.d - self.r) * 2 ** (-self.e)
            self._encoded = np.round(encoded).astype(dtype)
        self._bitmap = bitmap
        return (self._encoded, self._bitmap)

    def _determine_dtype(self):
        if self.n == 0 or self.n == 8:
            return ">u1"
        elif self.n == 16:
            return ">u2"
        elif self.n == 32:
            return ">u4"
        elif self.n == 64:
            return ">u8"
        else:
            raise RuntimeError("n other than 0, 8, 16, 32, and 64 is not supported")

    def write_sect5(self, f: BinaryIO) -> int:
        """Writes parameter data to the stream as Section 5 octet sequence."""
        _, _ = self.encode()
        num_of_values = self._len
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
        write(f, header)
        write(f, main_buf)
        write(f, template_buf)
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
        write(f, header)
        write(f, main_buf)
        write(f, bitmap)
        return sect_len

    def write_sect7(self, f: BinaryIO) -> int:
        """Writes encoded data to the stream as Section 7 octet sequence."""
        encoded, _ = self.encode()
        sect_len = SECT_HEADER_DTYPE.itemsize + encoded.nbytes
        header = create_sect_header(7, sect_len)
        write(f, header)
        write(f, encoded)
        return sect_len


def _get_parameters_simple_linear(data: np.ndarray, nbit: int):
    min = data.min()
    d = -ceil(np.log10((data.max() - min) / (2**nbit - 1)))
    r = min * 10**d
    return (r, d)


def _get_parameters_fixed_digit_linear(data: np.ndarray, decimals: int):
    inverse_precision = 10**decimals
    min = round(data.min() * inverse_precision)
    max = round(data.max() * inverse_precision)
    n_required = ceil(log2(max - min))
    n = _get_supported_nbit(n_required)
    return (min, n)


def _get_supported_nbit(n: int):
    if n > 32:
        return 64
    elif n > 16:
        return 32
    elif n > 8:
        return 16
    else:
        return 8


def create_bitmap(mask: NDArray[Shape["*"], Bool]) -> NDArray[Shape["*"], UInt8]:
    """Creates a bitmap octets corresponding to `mask`.

    In the input `mask`, True (1) must means that data is masked (missing).
    In the output bitmap, 0 means data is missing and 1 means data is present."""
    extra_len = len(mask) % 8
    n_pad = 0 if extra_len == 0 else 8 - extra_len
    array = np.pad(mask, (0, n_pad), constant_values=True).reshape(-1, 8)
    bits = np.packbits(~array, axis=-1).ravel()
    return bits


def _is_unique(data: np.ndarray):
    return (data == data[0]).all()
