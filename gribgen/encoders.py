import dataclasses
from typing import BinaryIO

import numpy as np

from gribgen.utils import SECT_HEADER_DTYPE, create_sect_header


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
        encoded = self.encode().tobytes()
        sect_len = SECT_HEADER_DTYPE.itemsize + len(encoded)
        header = create_sect_header(7, sect_len)
        f.write(header)
        f.write(encoded)
