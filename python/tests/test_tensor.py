import numpy as np
import tensor as ts
import pytest


def test_not_tensor_create_not_supported():
    with pytest.raises(ValueError) as e:
        ts.Tensor(shape=(2, ))
    expected = f"Tensor with dims other that 2D are not supported yet! Note that len(shape)=1"
    assert str(e.value) == expected

    with pytest.raises(ValueError) as e:
        ts.Tensor(shape=(2, 2, 2))
    expected = f"Tensor with dims other that 2D are not supported yet! Note that len(shape)=3"
    assert str(e.value) == expected


def test_tensor_shape():
    expected_shape = (2, 2)
    tensor = ts.Tensor(shape=expected_shape)
    assert isinstance(tensor.shape, tuple)
    assert tensor.shape == expected_shape


def test_tensor_matmul():
    t1 = ts.Tensor([[1, 2, 3], [4, 5, 6]])
    t2 = ts.Tensor([[7, 8], [9, 10], [11, 12]])
    t = t1 @ t2

    expected_array = np.array([[58., 64.], [139., 154.]])

    assert t.shape == expected_array.shape
    np.testing.assert_equal(t.numpy(), expected_array)
