import csv
from collections.abc import Iterator
from pathlib import Path
from typing import List, Tuple

import tensor.autograd.tensor_grad as tsg


class Dataset(Iterator):
    def __init__(self, path: Path, batch_size: int = 30, header: bool = True):
        self._batch_size: int = batch_size
        self._x: List[List[float]] = []
        self._y: List[int] = []
        self._begin: int = 0

        with path.open('r') as f:
            reader = csv.reader(f, delimiter='\t')
            if header:
                next(reader)
            for row in reader:
                self._x.append([float(v) for v in row[:2]])
                self._y.append(int(row[2]))

    def __len__(self):
        return len(self._x)

    def __next__(self) -> Tuple[tsg.Variable, tsg.Variable]:
        end = self._begin + self._batch_size
        if end > len(self):
            self._begin = 0
            raise StopIteration

        data = tsg.var(self._x[self._begin: end]), tsg.var(self._y[self._begin: end])
        self._begin += self._batch_size
        return data

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y: List[int]):
        self._y = y
