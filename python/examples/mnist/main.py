import time
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import tensor as ts
from tensor.autograd import Variable

from torchvision import datasets, transforms


class Net:
    def __init__(self):
        self.conv1 = ts.nn.Conv2D(1, 32, 3, 1, activation=ts.nn.Activation.RELU)
        self.conv2 = ts.nn.Conv2D(32, 64, 3, 1, activation=ts.nn.Activation.RELU)
        self.fc1 = ts.nn.Linear(9216, 128, activation=ts.nn.Activation.RELU)
        self.fc2 = ts.nn.Linear(128, 10)
        self.max_pool = ts.nn.MaxPool2D(2, 2)

    def forward(self, x: Variable) -> Variable:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = ts.autograd.reshape(x, [-1, 9216])
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def weights(self) -> List[List[ts.libtensor.GradHolderF]]:
        return [self.conv1.weights(), self.conv2.weights(), self.fc1.weights(), self.fc2.weights()]


def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=32)

    model = Net()
    loss_fn = ts.nn.CrossEntropyLoss()
    optimizer = ts.libtensor.SGD(0.01)
    for w in model.weights():
        optimizer.register_params(w)

    time_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        x = ts.autograd.var(np.moveaxis(data.numpy(), 1, -1))
        y = ts.autograd.var(target.numpy())

        output = model.forward(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            epoch = 1
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.value[0]))
    time_end = time.time()
    print(f"Training time: {time_end - time_start}")

    time_start = time.time()
    eval(model, test_loader, loss_fn)
    time_end = time.time()
    print(f"Evaluation time: {time_end - time_start}")


def eval(model: Net, test_loader: DataLoader, loss_fn: ts.nn.CrossEntropyLoss):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            x = ts.autograd.var(np.moveaxis(data.numpy(), 1, -1))
            y = ts.autograd.var(target.numpy())
            output = model.forward(x)
            test_loss += loss_fn(output, y).value[0]
            pred = ts.argmax(output.value)
            correct += np.sum(pred.numpy == y.value.numpy)

    example_num = len(test_loader.dataset)  # type: ignore
    test_loss /= example_num
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, example_num,
        100. * correct / example_num))


if __name__ == '__main__':
    train()
