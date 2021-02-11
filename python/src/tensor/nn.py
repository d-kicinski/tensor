from tensor import tensor as ts
import tensor.libtensor as _ts


def softmax(tensor: ts.Tensor) -> ts.Tensor:
    return ts.Tensor(_ts.softmax(tensor.data))
