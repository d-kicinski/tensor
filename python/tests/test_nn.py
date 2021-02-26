import tensor.tensor as ts
import numpy as np
from tensor.autograd.autograd import var
from tensor.nn import ReLU


def test_relu():
    relu = ReLU()
    tensor = var([[1., -1.], [-1., 1.]])
    forward = relu(tensor)
    expected_forward = ts.Tensor([[1., 0.], [0., 1.]])
    np.testing.assert_equal(forward.value.numpy, expected_forward.numpy)

    d_output = ts.Tensor([[2., 2.], [2., 2.]])
    relu.backward(d_output)
    expected_backward = ts.Tensor([[2., 0.], [0., 2.]])
    np.testing.assert_equal(tensor.grad.numpy, expected_backward.numpy)
