import numpy as np

from tensor import tensor as ts
import tensor.libtensor as _ts

from tensor.autograd.autograd import Op, Variable


def softmax(tensor: ts.Tensor) -> ts.Tensor:
    return ts.Tensor(_ts.softmax(tensor.data))


class ReLU(Op):
    def __init__(self):
        super().__init__()
        self._relu = None

    def forward(self, *inputs: Variable):
        tensor: Variable
        tensor = self._check_inputs(*inputs, num=1)
        self._inputs.extend([tensor])
        if self._relu is None:
            if tensor.value.dim == 2:
                self._relu = _ts.ReLU_f2()
            elif tensor.value.dim == 3:
                self._relu = _ts.ReLU_f3()
            else:
                raise ValueError(f"Incompatible input dim (dim={tensor.value.dim}) with ReLU op")
        value = self._relu(tensor.value.data)
        return Variable(ts.Tensor(value), self)

    def backward(self, *grads: ts.Tensor):
        d_output = self._check_grads(*grads, num=1)
        d_input = self._relu.backward(d_output.data)
        self._inputs[0].grad = ts.Tensor(d_input)


def relu(x: Variable):
    op = ReLU()
    return op(x)

