from typing import List

import numpy as np
import torch

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
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    model = Net()
    loss_fn = ts.nn.CrossEntropyLoss()
    optimizer = ts.libtensor.SGD(0.01)
    for w in model.weights():
        optimizer.register_params(w)

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


if __name__ == '__main__':
    train()
