import pytensor as pts


def test_create_empty_tensor():
    tensor = pts.TensorF2()
    assert tensor.data_size() == 0


def test_create_initialized_tensor():
    tensor = pts.TensorF2([2, 2])

    assert tensor.data_size() == 4


def test_random_tensor():
    tensor = pts.TensorF2().randn([2, 2])

    assert tensor.data_size() == 4
