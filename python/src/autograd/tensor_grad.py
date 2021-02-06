from abc import abstractmethod, ABCMeta
from typing import Optional, List, Iterable, Union, TypeVar

import numpy as np
import tensor as ts

T = TypeVar("T")
IterT = Union[T, Iterable[T]]


class Variable:

    def __init__(self, value: ts.Tensor, op: Optional["Op"] = None):
        self._value: ts.Tensor = value
        self._grad: ts.Tensor = ts.Tensor(np.full(value.shape, 1.0))
        self.op: Optional[Op] = op

    @property
    def value(self):
        return self._value

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    def __str__(self):
        return f"Variable({self.value.shape=}, {self._grad.shape=})"

    def __neg__(self) -> "Variable":
        var = Variable(-self._value, self.op)
        var._grad = self._grad
        return var

    def __matmul__(self, other: "Variable") -> "Variable":
        return dot(self, other)

    def __add__(self, other: "Variable") -> "Variable":
        return add(self, other)


class Op(metaclass=ABCMeta):

    def __init__(self):
        self._inputs = []

    @property
    def inputs(self) -> List[Variable]:
        return self._inputs

    @abstractmethod
    def forward(self, *args: Variable):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args: ts.Tensor):
        raise NotImplementedError

    @staticmethod
    def _check_inputs(*inputs: Variable, num: int) -> IterT[Variable]:
        if len(inputs) == num:
            if len(inputs) == 1:
                return inputs[0]
            else:
                return inputs
        else:
            raise ValueError(f"Incompatible input parameters. Length of inputs = {len(inputs)} ")

    @staticmethod
    def _check_grads(*grads: ts.Tensor, num: int) -> IterT[ts.Tensor]:
        if len(grads) == num:
            if len(grads) == 1:
                return grads[0]
            else:
                return grads
        else:
            raise ValueError(f"Incompatible grads parameters. Length of grads = {len(grads)} ")


class Add(Op):
    EXPECTED_INPUTS_LENGTH: int = 2
    EXPECTED_GRADS_LENGTH: int = 1

    def forward(self, *inputs: Variable) -> Variable:
        x, b = self._check_inputs(*inputs, num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        self._inputs.extend([x, b])

        if x.value.shape == b.value.shape:
            return Variable(x.value + b.value, self)
        elif b.value.shape[1] != x.value.shape[1] and b.value.shape[1] == 1:
            # TODO: do not use numpy to do this, do not cheat :^)
            b_broad = ts.Tensor(np.array(np.tile(b.value.numpy, (1, x.value.shape[1]))))
            return Variable(x.value + b_broad, self)
        else:
            raise ValueError(f"Add(Op): Unsupported input shapes! {x.value.shape}, {b.value.shape}")

    def backward(self, *grads: ts.Tensor):
        grad = self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        x = self._inputs[0]
        b = self._inputs[1]

        x.grad = grad

        if x.value.shape == b.value.shape:
            b.grad = grad
        elif b.value.shape[1] != x.value.shape[1] and b.value.shape[1] == 1:
            b.grad = ts.sum(grad, 0)
        else:
            raise ValueError(f"Add(Op): Unsupported input shapes! {x.value.shape}, {b.value.shape}")

    def __str__(self):
        return f"Add(x, y)"


class Dot(Op):
    EXPECTED_INPUTS_LENGTH: int = 2
    EXPECTED_GRADS_LENGTH: int = 1

    def forward(self, *inputs: Variable):
        a, b = self._check_inputs(*inputs, num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        self._inputs.extend([a, b])
        return Variable(a.value @ b.value, self)

    def backward(self, *grads: ts.Tensor):
        grad = self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        self._inputs[0].grad = grad @ self._inputs[1].value.T
        self._inputs[1].grad = self._inputs[0].value.T @ grad

    def __str__(self):
        return f"Dot(a, b)"


def dot(x: Variable, y: Variable) -> Variable:
    op = Dot()
    return op.forward(x, y)


def add(x: Variable, y: Variable) -> Variable:
    op = Add()
    return op.forward(x, y)


def traverse(var: Variable):
    if var.op:
        var.op.backward(var.grad)
        if inputs := var.op.inputs:
            for i in inputs:
                traverse(i)


def print_graph(var: Variable, prefix=""):
    delimiter = "    "

    def loop(v: Variable, p=""):
        p += delimiter
        print(p + str(v))
        if op := v.op:
            p += delimiter
            print(p + str(op))
            if inputs := op.inputs:
                for i in inputs:
                    print_graph(i, p)

    loop(var, prefix)


def main():
    x = Variable(ts.Tensor(
        [[1, 1],
         [2, 2],
         [3, 3]]
    ))

    w0 = Variable(ts.Tensor(
        [[4, 4, 4],
         [5, 5, 5]]
    ))

    b = Variable(ts.Tensor(
        [[4],
         [5]]
    ))

    w1 = Variable(ts.Tensor(
        [[1, 1],
         [2, 2],
         [3, 3]]
    ))

    y = w1 @ (w0 @ x + b)

    traverse(y)
    print_graph(y)


if __name__ == '__main__':
    main()
