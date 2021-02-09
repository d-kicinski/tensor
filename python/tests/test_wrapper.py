import libtensor as _ts
import pytest
import numpy as np


def test_matrix_f_init():
    tensor = _ts.MatrixF(2, 2)
    assert tensor.shape() == [2, 2]
    assert tensor.data_size() == 4


def test_matrix_f_getitem():
    tensor = _ts.MatrixF(2, 2)
    assert tensor[0, 0] == 0.
    assert tensor[0, 1] == 0.
    assert tensor[1, 0] == 0.
    assert tensor[1, 1] == 0.


def test_matrix_f_setitem():
    tensor = _ts.MatrixF(2, 2)
    assert tensor[1, 1] == 0.
    tensor[1, 1] = 1337.
    assert tensor[1, 1] == 1337.


def test_tensor_from_numpy():
    with pytest.raises(RuntimeError) as e:
        _ts.MatrixF(np.array([1, 2, 3]))  # trying to assign a int array
    assert str(e.value) == "Incompatible buffer format!"

    array = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    tensor = _ts.MatrixF(array)

    for i in range(tensor.shape()[0]):
        for j in range(tensor.shape()[1]):
            assert tensor[i, j] == array[i, j]


def test_numpy_from_tensor():
    tensor = _ts.MatrixF(5, 4)
    assert memoryview(tensor).shape == (5, 4)

    assert tensor[2, 3] == 0.

    tensor[2, 3] = 11.
    tensor[3, 2] = 7.
    assert tensor[2, 3] == 11.
    assert tensor[3, 2] == 7.

    array = np.array(tensor)
    assert array.shape == (5, 4)
    assert array[2, 3] == 11.
    assert array[3, 2] == 7.
    assert abs(array).sum() == 11 + 7

    array[2, 3] = 5.
    assert array[2, 3] == 5.


def test_cross_entropy_loss():
    loss = _ts.CrossEntropyLoss()
    logits = _ts.MatrixF(np.random.randn(300, 3).astype(np.float32))
    labels = _ts.VectorI(np.random.randint(0, 2, (300,)).astype(np.int32))

    loss_value = loss.forward(logits, labels)
    loss.backward()

    assert loss_value != 0.0
