from typing import Tuple

import pytensor as _ts


class Tensor:

    def __init__(self, *dims: int):
        if len(dims) != 2:
            print(dims)
            msg = f"Tensor with dims other that 2D are not supported yet! Note that {len(dims)=}"
            raise ValueError(msg)
        self._data = _ts.Tensor2F(*dims)
        self._shape: Tuple[int] = dims

    @property
    def shape(self):
        return self._shape

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"Tensor({self._shape[0]}, {self._shape[1]})"

