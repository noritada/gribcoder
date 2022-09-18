import dataclasses
from typing import BinaryIO

import numpy as np

from gribgen.utils import bytes_from_int


@dataclasses.dataclass
class SimplePackingEncoder:
    r: float
    e: int
    d: int
    n: int

    def data(self, data: np.ndarray):  # `-> Self` for Python >=3.11 (PEP 673)
        """Sets data to be encoded."""
        self._data = data
        return self

    def encode(self):
        """Packs and return as an `np.ndarray` with `n` bits occupied by each element."""
        if self._data is None:
            raise RuntimeError("data is not specified")
        dtype = self._determine_dtype()
        encoded = (self._data * 10**self.d - self.r) * 2 ** (-self.e)
        encoded = np.round(encoded).astype(dtype)
        return encoded

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

    def write_sect7(self, f: BinaryIO):
        """Writes encoded data to the stream as Section 7 octet sequence."""
        encoded = self.encode()
        encoded = np.frombuffer(encoded.tobytes(), dtype=np.uint8)
        sect_len = len(encoded) + 5
        f.write(bytes_from_int(sect_len, 4))
        f.write(bytes_from_int(0x07, 1))
        f.write(encoded)
