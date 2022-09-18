import struct

import numpy as np
from nptyping import NDArray, Shape, UInt8

SECT_HEADER_DTYPE = np.dtype(
    [
        ("sect_len", ">u4"),
        ("sect_num", "u1"),
    ]
)


def create_sect_header(num: int, length: int) -> np.ndarray:
    return np.array([(length, num)], dtype=SECT_HEADER_DTYPE)


def grib_int(num: int, byte_length: int) -> int:
    if num < 0:
        num = set_bit_one(-num, (byte_length * 8 - 1))
    return num


# There might be more direct and efficient solutions for the following conversions.


def bytes_from_int(num: int, length: int) -> NDArray[Shape["*"], UInt8]:
    if num < 0:
        num = set_bit_one(-num, (length * 8 - 1))
    bin = np.frombuffer(
        num.to_bytes(length, byteorder="big", signed=False), dtype=np.uint8
    )
    return bin


def set_bit_one(value: int, n: int) -> int:
    """Sets `n`th bit of `value` to 1."""
    return value | (1 << n)


def bytes_from_float(num: float, length: int) -> NDArray[Shape["*"], UInt8]:
    if length == 4:
        bin = np.frombuffer(struct.pack(">f", num), dtype=np.uint8)
    else:
        raise RuntimeError(f"unimplemented length: {length}")
    return bin
