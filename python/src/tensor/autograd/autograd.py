from __future__ import annotations

from abc import abstractmethod, ABCMeta
from typing import Optional, List, Iterable, Union, TypeVar

import numpy as np
from .. import tensor as ts

T = TypeVar("T")
IterT = Union[T, Iterable[T]]


class Variable:

    def __init__(self, value: ts.Tensor, op: Optional[Op] = None):
        self._value: ts.Tensor = value
        self._grad: ts.Tensor = ts.Tensor(np.full(value.shape, 1.0))
        self.op: Optional[Op] = op

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: ts.Tensor):
        self._value = value

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value: ts.Tensor):
        self._grad = value

    def backward(self):
        traverse(self)

    def __str__(self):
        return f"Variable"

    def __neg__(self) -> Variable:
        variable = Variable(-self._value, self.op)
        variable._grad = self._grad
        return variable

    def __matmul__(self, other: Variable) -> Variable:
        return matmul(self, other)

    def __add__(self, other: Variable) -> Variable:
        return add(self, other)


class Op(metaclass=ABCMeta):

    def __init__(self):
        self._inputs = []

    @property
    def inputs(self) -> List[Variable]:
        return self._inputs

    def __call__(self, *args: Variable):
        return self.forward(*args)

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
        x: Variable
        b: Variable

        x, b = self._check_inputs(*inputs, num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        self._inputs.extend([x, b])
        return Variable(x.value + b.value, self)

    def backward(self, *grads: ts.Tensor):
        grad = self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        x = self._inputs[0]
        b = self._inputs[1]

        x.grad = grad

        if x.value.shape == b.value.shape:
            b.grad = grad
        elif b.value.dim == 1 and x.value.shape[1] == b.value.shape[0]:
            b.grad = ts.sum(grad, 0)
        else:
            raise ValueError(f"Add(Op): Unsupported input shapes! {x.value.shape}, {b.value.shape}")

    def __str__(self):
        return f"Add"


class MatMul(Op):
    EXPECTED_INPUTS_LENGTH: int = 2
    EXPECTED_GRADS_LENGTH: int = 1

    def forward(self, *inputs: Variable):
        a: Variable
        b: Variable

        a, b = self._check_inputs(*inputs, num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        self._inputs.extend([a, b])
        return Variable(a.value @ b.value, self)

    def backward(self, *grads: ts.Tensor):
        grad = self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        self._inputs[0].grad = grad @ self._inputs[1].value.T
        self._inputs[1].grad = self._inputs[0].value.T @ grad

    def __str__(self):
        return f"MatMul"


class Log(Op):
    EXPECTED_INPUTS_LENGTH: int = 1
    EXPECTED_GRADS_LENGTH: int = 1

    def forward(self, *inputs: Variable):
        x: Variable

        x = self._check_inputs(*inputs, num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        self._inputs.extend([x])
        return Variable(ts.log(x.value), self)

    def backward(self, *grads: ts.Tensor):
        grad = self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        self._inputs[0].grad = self._inputs[0].value * grad

    def __str__(self):
        return f"Log"


class Reshape(Op):
    def __init__(self, shape: List[int]):
        super(Reshape, self).__init__()
        self._shape_after = shape

    def forward(self, *inputs: Variable):
        x: Variable
        x = self._check_inputs(*inputs, num=1)  # type: ignore
        if len(self._inputs) == 0:
            self._inputs.append(x)
        return Variable(x.value.reshape(self._shape_after), self)

    def backward(self, *grads: ts.Tensor):
        grad = self._check_grads(*grads, num=1)
        self._inputs[0].grad = grad.reshape(self._inputs[0].value.shape)

    def __str__(self):
        return f"Reshape"


def matmul(x: Variable, y: Variable) -> Variable:
    op = MatMul()
    return op(x, y)


def add(x: Variable, y: Variable) -> Variable:
    op = Add()
    return op(x, y)


def log(x: Variable) -> Variable:
    op = Log()
    return op(x)


def reshape(x: Variable, shape: List[int]) -> Variable:
    op = Reshape(shape)
    return op(x)


def var(*args, **kwargs) -> Variable:
    return Variable(ts.Tensor(*args), **kwargs)


def traverse(variable: Variable):
    if variable.op:
        variable.op.backward(variable.grad)
        if inputs := variable.op.inputs:
            for i in inputs:
                traverse(i)


def print_graph(variable: Variable, prefix=""):
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

    loop(variable, prefix)
