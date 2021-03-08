from tensor import tensor as ts
import tensor.libtensor as _ts
from tensor.libtensor import Activation
from tensor.autograd.autograd import Op, Variable


class Conv2D(Op):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 activation: _ts.Activation = _ts.Activation.NONE, use_bias: bool = True):
        super().__init__()
        self._layer = _ts.Conv2D(in_channels, out_channels, kernel_size, stride, activation,
                                 use_bias)

    def forward(self, *inputs: Variable):
        tensor: Variable
        tensor = self._check_inputs(*inputs, num=1)
        if len(self._inputs) == 0:
            self._inputs.append(tensor)
        value = self._layer(tensor.value.data)
        return Variable(ts.Tensor(value), self)

    def backward(self, *grads: ts.Tensor):
        d_output = self._check_grads(*grads, num=1)
        d_input = self._layer.backward(d_output.data)
        self._inputs[0].grad = ts.Tensor(d_input)


class Linear(Op):
    def __init__(self, dim_in: int, dim_out: int, activation: _ts.Activation = _ts.Activation.NONE):
        super().__init__()
        self._layer = _ts.FeedForward(dim_in, dim_out, activation)

    def forward(self, *inputs: Variable):
        tensor: Variable
        tensor = self._check_inputs(*inputs, num=1)
        if len(self._inputs) == 0:
            self._inputs.append(tensor)
        value = self._layer(tensor.value.data)
        return Variable(ts.Tensor(value), self)

    def backward(self, *grads: ts.Tensor):
        d_output = self._check_grads(*grads, num=1)
        d_input = self._layer.backward(d_output.data)
        self._inputs[0].grad = ts.Tensor(d_input)


class ReLU(Op):
    def __init__(self):
        super().__init__()
        self._relu = None

    def forward(self, *inputs: Variable):
        tensor: Variable
        tensor = self._check_inputs(*inputs, num=1)
        if len(self._inputs) == 0:
            self._inputs.append(tensor)
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


def softmax(tensor: ts.Tensor) -> ts.Tensor:
    return ts.Tensor(_ts.softmax(tensor.data))
