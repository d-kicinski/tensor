from . import tensor as ts
from . import libtensor as _ts
from .autograd import Op, Variable


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


class MaxPool2D(Op):
    def __init__(self, kernel_size: int, stride: int):
        super(MaxPool2D, self).__init__()
        self._layer = _ts.MaxPool2D(kernel_size, stride)

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


class CrossEntropyLoss(Op):
    EXPECTED_INPUTS_LENGTH: int = 2
    EXPECTED_GRADS_LENGTH: int = 1

    def __init__(self):
        super().__init__()
        self._loss = _ts.CrossEntropyLoss()

    def forward(self, *inputs: Variable):
        logits: Variable
        labels: Variable

        logits, labels = self._check_inputs(*inputs,
                                            num=self.EXPECTED_INPUTS_LENGTH)  # type: ignore
        if len(self._inputs) == 0:
            self._inputs.extend([logits, labels])

        loss_value = self._loss.forward(logits.value.data, labels.value.data)
        return Variable(ts.Tensor(loss_value), self)

    def backward(self, *grads: ts.Tensor):
        self._check_grads(*grads, num=self.EXPECTED_GRADS_LENGTH)
        grad: _ts.MatrixF = self._loss.backward()
        self.inputs[0].grad = ts.Tensor(grad)

    def __str__(self):
        return f"CrossEntropyLoss"


def relu(x: Variable):
    op = ReLU()
    return op(x)


def softmax(tensor: ts.Tensor) -> ts.Tensor:
    return ts.Tensor(_ts.softmax(tensor.data))


def cross_entropy_loss(y: Variable, labels: Variable) -> Variable:
    loss_fn = CrossEntropyLoss()
    return loss_fn(y, labels)
