import dataclasses

import numpy as np


@dataclasses.dataclass
class SimplePackingEncoder:
    r: float
    e: int
    d: int
    n: int

    def encode(self, data: np.ndarray):
        """Packs and return given `data` as an `np.ndarray` with `n` bits occupied by each element."""
        dtype = self._determine_dtype()
        encoded = (data * 10**self.d - self.r) * 2 ** (-self.e)
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
