from __future__ import annotations
from typing import Optional, Callable, Tuple

from numpy.typing import ArrayLike
from mnist import MNIST
import tensor as ts
import numpy as np


class MNISTDataset:

    def __init__(self, root: str, batch_size: int,
                 train: bool = True,
                 transform: Optional[Callable] = None) -> None:
        loader = MNIST(root, return_type="numpy")
        data, target = loader.load_training() if train else loader.load_testing()

        self._data = data.reshape(-1, 1, 28, 28).astype(np.float32)
        self._target = target.astype(np.int32)
        self._transform = transform
        self._n: int = 0
        self._batch_size: int = batch_size

    def __len__(self) -> int:
        return len(self._data) // self._batch_size

    def __iter__(self) -> MNISTDataset:
        self._n = 0
        return self

    def __next__(self) -> Tuple[ts.Tensor, ts.Tensor]:
        if self._n + 1 == len(self._data) // self._batch_size:
            raise StopIteration

        images = self._get_chunk(self._data)
        labels = self._get_chunk(self._target)
        self._n += 1

        if self._transform is not None:
            images = self._transform(images)

        return ts.Tensor(images), ts.Tensor(labels)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def example_num(self) -> int:
        return len(self._data)

    def _get_chunk(self, array: ArrayLike) -> ArrayLike:
        return array[self._batch_size * self._n: self._batch_size * (self._n + 1)]
