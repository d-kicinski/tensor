import tensor.autograd.tensor_grad as tsg
import tensor.tensor as ts
import numpy as np


def main():
    x = tsg.Variable(ts.Tensor(np.random.randn(300, 2)))

    w0 = tsg.Variable(ts.Tensor(np.random.randn(2, 100)))
    b0 = tsg.Variable(ts.Tensor(np.random.randn(100)))

    w1 = tsg.Variable(ts.Tensor(np.random.randn(100, 3)))

    y = (x @ w0 + b0) @ w1

    y.backward()
    tsg.print_graph(y)


if __name__ == '__main__':
    main()
