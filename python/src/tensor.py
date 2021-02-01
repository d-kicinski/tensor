from typing import Optional, Union, Sequence, List

import pytensor as _ts
import numpy as np

ArrayT = Union[_ts.Tensor2F, np.array, List[List[float]]]


class Tensor:

    def __init__(self,
                 array: Optional[ArrayT] = None,
                 shape: Optional[Sequence[int]] = None):

        self._data: _ts.Tensor2F
        self._shape: Sequence[int]

        if array is None and shape is None:
            raise ValueError("either array or dims should be passed to initialize object")

        if isinstance(array, _ts.Tensor2F):
            self._data = array
        elif isinstance(array, np.ndarray):
            self._check_shape(array.shape)
            self._data = _ts.Tensor2F(array.astype(np.float32))
        elif isinstance(array, List):
            arr = np.array(array)
            self._check_shape(arr.shape)
            self._data = _ts.Tensor2F(arr.astype(np.float32))
        elif array is not None:
            raise ValueError(f"array type {type(array)} is not supported")

        if shape:
            self._check_shape(shape)
            self._data: _ts.Tensor2F = _ts.Tensor2F(*shape)

        self._shape = tuple(self._data.shape())

    def _check_shape(self, shape: Sequence[int]):
        if len(shape) != 2:
            msg = f"Tensor with dims other that 2D are not supported yet! Note that {len(shape)=}"
            raise ValueError(msg)

    @property
    def shape(self):
        return self._shape

    def numpy(self):
        return np.array(self._data)

    def __add__(self, other: "Tensor") -> "Tensor":
        return Tensor(_ts.add(self._data, other._data))

    def __mul__(self, other: float) -> "Tensor":
        return Tensor(_ts.multiply(self._data, other))

    def __rmul__(self, other: float):
        return self * other

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return Tensor(_ts.dot(self._data, other._data))

    def __str__(self) -> str:
        return f"Tensor({self._shape[0]}, {self._shape[1]})"


def log(tensor: Tensor) -> Tensor:
    return Tensor(_ts.log(tensor._data))


def pow(tensor: Tensor, p: int) -> Tensor:
    return Tensor(_ts.pow(tensor._data, p))


def exp(tensor: Tensor) -> Tensor:
    return Tensor(_ts.exp(tensor._data))
