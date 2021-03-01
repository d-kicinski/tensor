from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Optional, Union, Sequence, List, Tuple, Any, Type, Callable, Final

import tensor.libtensor as _ts
import numpy as np
from numpy.typing import ArrayLike

DataT = Union[_ts.MatrixF, _ts.MatrixI, _ts.VectorF, _ts.VectorI]
ArrayT = Union[DataT, ArrayLike]
ScalarT = Number
NumpyT = Union[np.int32, np.float32]


@dataclass
class _NumberWrapper:
    data: Number
    dim: Final[int] = 0


def _is_instance_of_tensor(o: DataT) -> bool:
    types = [_ts.MatrixF, _ts.MatrixI, _ts.VectorF, _ts.VectorI]
    for t in types:
        if isinstance(o, t):
            return True
    return False


def _map_dim_and_type_to_tensor(dim: int, t: ArrayLike) -> Type[DataT]:
    dim_map = {1: "Vector", 2: "Matrix"}
    try:
        name = dim_map[dim]
    except KeyError:
        raise ValueError(f"Dim {dim} is not supported!")

    if t in (np.float32, float):
        name += "F"
    elif t in (np.int32, int):
        name += "I"
    else:
        raise ValueError(f"Type {t} is not supported!")

    return getattr(_ts, name)


def _numpy_downcast(array: np.array) -> np.array:
    dtype = array.dtype
    if dtype in [np.int32, np.float32]:
        return array
    elif dtype == np.int64:
        return array.astype(np.int32)
    elif dtype == np.float64:
        return array.astype(np.float32)
    else:
        raise ValueError(f"Array type {dtype} is not supported!")


def _map_ts_to_type(data: DataT) -> Type:
    if type(data) in [_ts.VectorI, _ts.MatrixI]:
        return int
    elif type(data) in [_ts.VectorF, _ts.MatrixF]:
        return float
    else:
        raise ValueError(f"Incompatible data type {type(data)}")


def _check_shape(shape: Sequence[int]):
    if len(shape) not in [1, 2]:
        msg = f"Tensor with dims other that 2D are not supported yet! Note that {len(shape)=}"
        raise ValueError(msg)


def _dispatch_native_method(prefix: str, dtype: Type, *dims: int) -> Callable:
    type_id = "i" if dtype is int else "f"
    dim_map = {0: "", 1: "vector", 2: "matrix", 3: "tensor3"}

    name = f"{prefix}"
    for dim in dims:
        name += f"_{dim_map[dim]}{type_id}"
    return getattr(_ts, name)


class Tensor:
    def __init__(self,
                 array: Optional[Union[ArrayT, ScalarT]] = None,
                 shape: Optional[Sequence[int]] = None):

        self._data: DataT
        self._data_type: Union[int, float]
        self._shape: Sequence[int]

        if array is None and shape is None:
            raise ValueError("either array or dims should be passed to initialize object")

        array32: np.array

        if _is_instance_of_tensor(array):
            self._data = array
        elif isinstance(array, np.ndarray):
            array32 = _numpy_downcast(array)
            self._data = _map_dim_and_type_to_tensor(array32.ndim, array32.dtype)(array32)
        elif isinstance(array, List):
            array32 = _numpy_downcast(np.array(array))
            self._data = _map_dim_and_type_to_tensor(array32.ndim, array32.dtype)(array32)
        elif isinstance(array, ScalarT):
            array32 = _numpy_downcast(np.array([array]))
            self._data = _map_dim_and_type_to_tensor(array32.ndim, array32.dtype)(array32)
        elif array is not None:
            raise ValueError(f"Array type {type(array)} is not supported~")

        if shape:
            _check_shape(shape)
            self._data = _map_dim_and_type_to_tensor(len(shape), float)(*shape)

        self._shape = tuple(self._data.shape())
        self._data_type = _map_ts_to_type(self._data)

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._data_type

    @property
    def data(self):
        return self._data

    @property
    def T(self):
        return Tensor(_ts.transpose(self._data))

    @property
    def numpy(self):
        return np.array(self._data)

    def __getitem__(self, item: Union[Tuple[int, int], int]) -> float:
        if isinstance(item, tuple):
            if len(item) != self.dim:
                raise IndexError(
                    f"Index {item} is not compatible with Tensor with shape {self.shape}")
            return self._data[item]
        elif isinstance(item, int):
            return self._data[item]
        else:
            raise ValueError(f"Unsupported index: {item}")

    def __add__(self, other: Tensor) -> Tensor:
        other = _NumberWrapper(other) if isinstance(other, Number) else other
        method = _dispatch_native_method("add", self.dtype, self.dim, other.dim)
        return Tensor(method(self._data, other.data))

    def __mul__(self, other: Union[float, Tensor]) -> Tensor:
        other = _NumberWrapper(other) if isinstance(other, Number) else other
        method = _dispatch_native_method("multiply", self.dtype, self.dim, other.dim)
        return Tensor(method(self._data, other.data))

    def __rmul__(self, other: Union[float, Tensor]) -> Tensor:
        return self * other

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(_ts.dot(self._data, other._data))

    def __str__(self) -> str:
        return f"Tensor({self.shape})"


def log(tensor: Tensor) -> Tensor:
    return Tensor(_ts.log(tensor.data))


def pow(tensor: Tensor, p: int) -> Tensor:
    return Tensor(_ts.pow(tensor.data, p))


def exp(tensor: Tensor) -> Tensor:
    return Tensor(_ts.exp(tensor.data))


def sum(tensor: Tensor, axis: int) -> Tensor:
    return Tensor(_ts.sum(tensor.data, axis))


def argmax(tensor: Tensor) -> Tensor:
    if tensor.dtype == int:
        return Tensor(_ts.argmax_i(tensor.data))
    elif tensor.dtype == float:
        return Tensor(_ts.argmax_f(tensor.data))
    else:
        raise ValueError(f"Incompatible tensor dtype {tensor.dtype}")
