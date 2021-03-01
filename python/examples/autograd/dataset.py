import csv
import random
from collections.abc import Iterator
from pathlib import Path
from typing import List, Tuple, Optional

import tensor.autograd.autograd as tsg

PointFloat = Tuple[float, float]


class Dataset(Iterator):
    def __init__(self, path: Optional[Path] = None, batch_size: int = 30, header: bool = True):
        self._batch_size: int = batch_size
        self._x: List[PointFloat] = []
        self._y: List[int] = []
        self._begin: int = 0

        if path:
            self._x, self._y = Dataset.load(path, header)

    def __len__(self):
        return len(self._x)

    def __next__(self) -> Tuple[tsg.Variable, tsg.Variable]:
        end = self._begin + self._batch_size
        if end > len(self):
            end = len(self)

        if self._begin >= len(self):
            self._begin = 0
            raise StopIteration

        data = tsg.var(self._x[self._begin: end]), tsg.var(self._y[self._begin: end])
        self._begin += self._batch_size
        return data

    def shuffle(self):
        zipped = list(zip(self._x, self._y))
        random.shuffle(zipped)
        self._x, self._y = zip(*zipped)
        self._x = list(self._x)
        self._y = list(self._y)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x: List[PointFloat]):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y: List[int]):
        self._y = y

    @staticmethod
    def load(path: Path, header: bool = True) -> Tuple[List[PointFloat], List[int]]:
        points: List[PointFloat] = []
        labels: List[int] = []
        with path.open('r') as f:
            reader = csv.reader(f, delimiter='\t')
            if header:
                next(reader)
            for row in reader:
                points.append((float(row[0]), float(row[1])))
                labels.append(int(row[2]))
        return points, labels
