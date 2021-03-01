from pathlib import Path
from typing import List

from tqdm import tqdm

import tensor.autograd.autograd as tsg
from tensor import nn
from tensor import tensor as ts
import numpy as np

from dataset import Dataset
from visualization import visualize

TRAIN_DATASET_PATH = Path("resources/train_planar_data.tsv")
TEST_DATASET_PATH = Path("resources/test_planar_data.tsv")


class Model:
    def __init__(self):
        self.w0 = tsg.var(np.random.randn(2, 100))
        self.b0 = tsg.var(np.random.randn(100))
        self.w1 = tsg.var(np.random.randn(100, 3))

    def __call__(self, x: tsg.Variable):
        return nn.relu(x @ self.w0 + self.b0) @ self.w1

    def update(self, lr: float = 1e-3):
        self.w0.value += -lr * self.w0.grad
        self.b0.value += -lr * self.b0.grad
        self.w1.value += -lr * self.w1.grad


def train(model: Model, dataset: Dataset, epochs: int = 100) -> Model:
    for i_epoch in range(epochs):
        for x, labels in dataset:
            y = model(x)
            loss = tsg.cross_entropy_loss(y, labels)
            loss.backward()
            model.update(1e-1)

        if i_epoch % 100 == 0:
            print(f"[{i_epoch + 1}/{epochs}] loss: {loss.value.data[0]}")

    print(f"[{i_epoch + 1}/{epochs}] loss: {loss.value.data[0]}")
    return model


def label_test_dataset(model: Model, dataset: Dataset) -> List[int]:
    labels: List[int] = []
    for x, _ in tqdm(dataset, desc="Labeling"):
        y = ts.argmax(nn.softmax(model(x).value))
        labels.extend(y.numpy.tolist())
    return labels


def main():
    dataset = Dataset(TRAIN_DATASET_PATH, batch_size=10)
    dataset.shuffle()

    print("Training model...")
    model: Model = train(Model(), dataset)
    print("Done!")

    dataset_test = Dataset(TEST_DATASET_PATH, batch_size=500)
    labels: List[int] = label_test_dataset(model, dataset_test)
    dataset_test.y = labels

    visualize(dataset, dataset_test)


if __name__ == '__main__':
    main()
