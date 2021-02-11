from collections.abc import Iterator
from pathlib import Path
from typing import Tuple, List
import csv

import tensor.autograd.tensor_grad as tsg
import numpy as np

TRAIN_DATASET_PATH = Path("resources/train_planar_data.tsv")


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


def main():
    epochs = 100
    batch_size = 300
    dataset = Dataset(TRAIN_DATASET_PATH, batch_size=batch_size)
    batch_num = len(dataset) // batch_size
    print(f"examples: {len(dataset)}")
    print(f"batches: {batch_num}")

    w0 = tsg.var(np.random.randn(2, 100))
    b0 = tsg.var(np.random.randn(100))
    w1 = tsg.var(np.random.randn(100, 3))
    lr: float = 1e-0

    for i_epoch in range(epochs):
        for x, labels in dataset:
            y = (x @ w0 + b0) @ w1

            loss = tsg.cross_entropy_loss(y, labels)
            loss.backward()

            w0.value += -lr * w0.grad
            b0.value += -lr * b0.grad
            w1.value += -lr * w1.grad

        if i_epoch % 10:
            print(f"[{i_epoch}/{epochs}] loss: {loss.value.data[0]}")


if __name__ == '__main__':
    main()
