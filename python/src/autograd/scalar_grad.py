from abc import abstractmethod, ABCMeta
from math import log
from typing import Optional, List, Iterable, Union, TypeVar

T = TypeVar("T")
IterT = Union[T, Iterable[T]]


class Variable:

    def __init__(self, value: float, op: Optional["Op"] = None):
        self._value: float = value
        self._grad: float = 1.0
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
        return f"Variable({self.value=}, {self._grad=})"

    def __neg__(self) -> "Variable":
        var = Variable(-self._value, self.op)
        var._grad = self._grad
        return var

    def __mul__(self, other: "Variable") -> "Variable":
        return multiply(self, other)

    def __add__(self, other: "Variable") -> "Variable":
        return add(self, other)

    def __sub__(self, other: "Variable") -> "Variable":
        return sub(self, other)

    def __pow__(self, p: "Variable", modulo=None) -> "Variable":
        return power(self, p)


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
    def backward(self, *args: float):
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
    def _check_grads(*grads: float, num: int) -> IterT[float]:
        if len(grads) == num:
            if len(grads) == 1:
                return grads[0]
            else:
                return grads
        else:
            raise ValueError(f"Incompatible grads parameters. Length of grads = {len(grads)} ")


class Multiply(Op):
    EXPECTED_INPUTS_LENGTH: int = 2
    EXPECTED_GRADS_LENGTH: int = 1

    def forward(self, *inputs: Variable) -> Variable:
        x, y = self._check_inputs(*inputs, num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        self._inputs.extend([x, y])
        ret = Variable(x.value * y.value, self)
        return ret

    def backward(self, *grads: float):
        grad = self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        self._inputs[0].grad = self._inputs[1].value * grad
        self._inputs[1].grad = self._inputs[0].value * grad

    def __str__(self):
        return f"Multiply(x, y)"


class Add(Op):
    EXPECTED_INPUTS_LENGTH: int = 2
    EXPECTED_GRADS_LENGTH: int = 1

    def forward(self, *inputs: Variable) -> Variable:
        x, y = self._check_inputs(*inputs, num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        self._inputs.extend([x, y])
        ret = Variable(x.value + y.value)
        ret.op = self
        return ret

    def backward(self, *grads: float):
        grad = self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        self._inputs[0].grad = grad
        self._inputs[1].grad = grad

    def __str__(self):
        return f"Add(x, y)"


class Sub(Op):
    EXPECTED_INPUTS_LENGTH: int = 2
    EXPECTED_GRADS_LENGTH: int = 1

    def forward(self, *inputs: Variable) -> Variable:
        x, y = self._check_inputs(*inputs, num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        self._inputs.extend([x, y])
        ret = Variable(x.value - y.value)
        ret.op = self
        return ret

    def backward(self, *grads: float):
        grad = self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        self._inputs[0].grad = grad
        self._inputs[1].grad = -grad  # type: ignore

    def __str__(self):
        return f"Sub(x, y)"


class Pow(Op):
    EXPECTED_INPUTS_LENGTH: int = 2
    EXPECTED_GRADS_LENGTH: int = 1

    def forward(self, *inputs: Variable) -> Variable:
        x, p = self._check_inputs(*inputs, num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        self._inputs.extend([x, p])
        ret = Variable(x.value ** p.value, self)
        return ret

    def backward(self, *grads: float):
        grad = self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        x, p = self._inputs[0].value, self._inputs[1].value
        self._inputs[0].grad = p * x ** (p - 1) * grad
        self._inputs[1].grad = x ** p * log(x) * grad

    def __str__(self):
        return f"Pow(x, p)"


def multiply(x: Variable, y: Variable) -> Variable:
    op = Multiply()
    return op.forward(x, y)


def add(x: Variable, y: Variable) -> Variable:
    op = Add()
    return op.forward(x, y)


def sub(x: Variable, y: Variable) -> Variable:
    op = Sub()
    return op.forward(x, y)


def power(x: Variable, p: Variable) -> Variable:
    op = Pow()
    return op.forward(x, p)


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
    x = Variable(2)
    w = Variable(3)
    b = Variable(4)
    p = Variable(3)
    c = Variable(1)

    y = x * w ** (p - c) + b

    traverse(y)
    print_graph(y)


if __name__ == '__main__':
    main()
