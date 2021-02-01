import tensor as ts
import pytest


def test_not_tensor_create_not_supported():
    with  pytest.raises(ValueError) as e:
        tensor = ts.Tensor(2)

    expected = f"Tensor with dims other that 2D are not supported yet! Note that len(dims)=1"
    assert str(e.value) == expected

    with  pytest.raises(ValueError) as e:
        tensor = ts.Tensor(2, 2, 2)

    expected = f"Tensor with dims other that 2D are not supported yet! Note that len(dims)=3"
    assert str(e.value) == expected


def test_tensor_create():
    tensor = ts.Tensor(2, 2)
    assert tensor is not None


def test_tensor_shape():
    expected_shape = (2, 2)
    tensor = ts.Tensor(*expected_shape)
    assert isinstance(tensor.shape, tuple)
    assert tensor.shape == expected_shape
