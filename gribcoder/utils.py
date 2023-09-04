from typing import BinaryIO

import numpy as np

SECT_HEADER_DTYPE = np.dtype(
    [
        ("sect_len", ">u4"),
        ("sect_num", "u1"),
    ]
)


def create_sect_header(num: int, length: int) -> np.ndarray:
    return np.array([(length, num)], dtype=SECT_HEADER_DTYPE)


def grib_signed(num: int, byte_length: int) -> int:
    if num < 0:
        num = set_bit_one(-num, (byte_length * 8 - 1))
    return num


def set_bit_one(value: int, n: int) -> int:
    """Sets `n`th bit of `value` to 1."""
    return value | (1 << n)


def write(f: BinaryIO, array: np.ndarray) -> int:
    f.write(array)
    return array.nbytes
