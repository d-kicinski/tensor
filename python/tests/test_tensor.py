import numpy as np
import tensor as ts
import pytest


def test_not_tensor_create_not_supported():
    with pytest.raises(ValueError) as e:
        ts.Tensor(shape=(2,))
    expected = f"Tensor with dims other that 2D are not supported yet! Note that len(shape)=1"
    assert str(e.value) == expected

    with pytest.raises(ValueError) as e:
        ts.Tensor(shape=(2, 2, 2))
    expected = f"Tensor with dims other that 2D are not supported yet! Note that len(shape)=3"
    assert str(e.value) == expected


def test_tensor_scalar():
    scalar = ts.Tensor(7)
    assert scalar.shape == (1, 1)
    assert scalar[0, 0] == 7


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


def test_tensor_add():
    t1 = ts.Tensor([[1, 1], [2, 2]])
    t2 = ts.Tensor([[2, 2], [1, 1]])
    t = t1 + t2

    expected_array = np.array([[3, 3], [3, 3]])

    assert t.shape == expected_array.shape
    np.testing.assert_equal(t.numpy(), expected_array)


def test_tensor_mul():
    t1 = ts.Tensor([[1, 2], [3, 4]])
    t = 2 * t1

    expected_array = np.array([[2, 4], [6, 8]])

    assert t.shape == expected_array.shape
    np.testing.assert_equal(t.numpy(), expected_array)


def test_log():
    t1 = ts.Tensor([[2, 2], [4, 4]])
    t = ts.log(t1)

    expected_array = np.log(t1.numpy())

    np.testing.assert_almost_equal(t.numpy(), expected_array)


def test_exp():
    t1 = ts.Tensor([[2, 2], [4, 4]])
    t = ts.exp(t1)

    expected_array = np.exp(t1.numpy())

    np.testing.assert_almost_equal(t.numpy(), expected_array, decimal=6)


def test_pow():
    t1 = ts.Tensor([[2, 2], [4, 4]])
    t = ts.pow(t1, 4)

    expected_array = np.power(t1.numpy(), 4)

    np.testing.assert_almost_equal(t.numpy(), expected_array)


def test_tensor_get_item():
    tensor = ts.Tensor([[1, 2], [3, 4]])
    assert tensor[0, 0] == 1
    assert tensor[0, 1] == 2
    assert tensor[1, 0] == 3
    assert tensor[1, 1] == 4


def test_tensor_get_item_scalar():
    tensor = ts.Tensor(7)
    assert tensor[0] == 7


def test_tensor_transpose():
    tensor = ts.Tensor(np.array([[1, 2, 3], [1, 2, 3]]))
    tensor_t = tensor.T

    expected_array = np.array([[1, 1], [2, 2], [3, 3]])

    np.testing.assert_equal(tensor_t.numpy(), expected_array)
