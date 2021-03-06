from typing import Any, List, Optional, Tuple, TypeVar

from typing import overload

buffer = TypeVar("buffer")

def add_matrixf_matrixf(arg0: MatrixF, arg1: MatrixF) -> MatrixF: ...
def add_matrixf_vectorf(arg0: MatrixF, arg1: VectorF) -> MatrixF: ...
def add_matrixi_matrixi(arg0: MatrixI, arg1: MatrixI) -> MatrixI: ...
def add_matrixi_vectori(arg0: MatrixI, arg1: VectorI) -> MatrixI: ...
def add_vectorf_vectorf(arg0: VectorF, arg1: VectorF) -> VectorF: ...
def add_vectori_vectori(arg0: VectorI, arg1: VectorI) -> VectorI: ...
def argmax_f(arg0: MatrixF) -> VectorI: ...
def argmax_i(arg0: MatrixI) -> VectorI: ...
@overload
def dot(arg0: VectorF, arg1: VectorF) -> float: ...
@overload
def dot(arg0: MatrixF, arg1: VectorF, arg2: bool) -> VectorF: ...
@overload
def dot(A: MatrixF, B: MatrixF, A_T: bool = ..., B_T: bool = ...) -> MatrixF: ...
@overload
def dot(arg0: Tensor3F, arg1: MatrixF) -> Tensor3F: ...
@overload
def dot(*args, **kwargs) -> Any: ...
def exp(arg0: MatrixF) -> MatrixF: ...
def get(arg0: MatrixF, arg1: MatrixF) -> MatrixF: ...
def log(arg0: MatrixF) -> MatrixF: ...
def log_softmax(arg0: MatrixF) -> MatrixF: ...
def multiply_matrixf_f(arg0: MatrixF, arg1: float) -> MatrixF: ...
def multiply_matrixf_matrixf(arg0: MatrixF, arg1: MatrixF) -> MatrixF: ...
def multiply_vectorf_f(arg0: VectorF, arg1: float) -> VectorF: ...
def multiply_vectorf_vectorf(arg0: VectorF, arg1: VectorF) -> VectorF: ...
def outer_product(arg0: VectorF, arg1: VectorF) -> MatrixF: ...
def pow(arg0: MatrixF, arg1: int) -> MatrixF: ...
def softmax(arg0: MatrixF) -> MatrixF: ...
def sum(arg0: MatrixF, arg1: int) -> VectorF: ...
def transpose(arg0: MatrixF) -> MatrixF: ...

class Activation:
    NONE: Any = ...
    RELU: Any = ...
    __entries: Any = ...
    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> Any: ...
    @property
    def __doc__(self) -> Any: ...
    @property
    def __members__(self) -> Any: ...

class LayerBase:
    def __init__(self) -> None: ...
    def parameters(self) -> List[DataHolderF]: ...
    def register_parameter(self, arg0: DataHolderF) -> None: ...
    def register_parameters(self, arg0: List[DataHolderF]) -> None: ...

class Conv2D(LayerBase):
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int, arg4: Activation, arg5: bool) -> None: ...
    def backward(self, arg0: Tensor4F) -> Tensor4F: ...
    def bias(self) -> Optional[Variable1F]: ...
    def forward(self, arg0: Tensor4F) -> Tensor4F: ...
    def parameters(self) -> List[DataHolderF]: ...
    def weight(self) -> Variable2F: ...
    def weights(self) -> List[GradHolderF]: ...
    def __call__(self, arg0: Tensor4F) -> Tensor4F: ...

class CrossEntropyLoss:
    def __init__(self) -> None: ...
    def backward(self) -> MatrixF: ...
    def forward(self, arg0: MatrixF, arg1: VectorI) -> float: ...
    def __call__(self, arg0: MatrixF, arg1: VectorI) -> float: ...

class DataHolderF:
    def __init__(self) -> None: ...

class DataHolderI:
    def __init__(self) -> None: ...

class FeedForward(LayerBase):
    def __init__(self, arg0: int, arg1: int, arg2: Activation) -> None: ...
    def backward(self, arg0: MatrixF) -> MatrixF: ...
    def bias(self) -> Variable1F: ...
    def forward(self, arg0: MatrixF) -> MatrixF: ...
    def parameters(self) -> List[DataHolderF]: ...
    def weight(self) -> Variable2F: ...
    def weights(self) -> List[GradHolderF]: ...
    def __call__(self, arg0: MatrixF) -> MatrixF: ...

class GradHolderF:
    def __init__(self) -> None: ...
    def grad(self) -> DataHolderF: ...
    def tensor(self) -> DataHolderF: ...

class GradHolderI:
    def __init__(self) -> None: ...
    def grad(self) -> DataHolderI: ...
    def tensor(self) -> DataHolderI: ...

class MatrixF(DataHolderF):
    @overload
    def __init__(self, arg0: int, arg1: int) -> None: ...
    @overload
    def __init__(self, arg0: buffer) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def data_size(self) -> int: ...
    def reshape1(self, *args, **kwargs) -> Any: ...
    def reshape2(self, arg0: List[int[2]]) -> MatrixF: ...
    def reshape3(self, arg0: List[int[3]]) -> Tensor3F: ...
    def reshape4(self, arg0: List[int[4]]) -> Tensor4F: ...
    def shape(self) -> List[int[2]]: ...
    def __getitem__(self, arg0: Tuple[int,int]) -> float: ...
    def __setitem__(self, arg0: Tuple[int,int], arg1: float) -> None: ...

class MatrixI(DataHolderI):
    @overload
    def __init__(self, arg0: int, arg1: int) -> None: ...
    @overload
    def __init__(self, arg0: buffer) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def data_size(self) -> int: ...
    def reshape1(self, *args, **kwargs) -> Any: ...
    def reshape2(self, arg0: List[int[2]]) -> MatrixI: ...
    def reshape3(self, arg0: List[int[3]]) -> Tensor3I: ...
    def reshape4(self, arg0: List[int[4]]) -> Tensor4I: ...
    def shape(self) -> List[int[2]]: ...
    def __getitem__(self, arg0: Tuple[int,int]) -> int: ...
    def __setitem__(self, arg0: Tuple[int,int], arg1: int) -> None: ...

class MaxPool2D:
    def __init__(self, arg0: int, arg1: int) -> None: ...
    def backward(self, arg0: Tensor4F) -> Tensor4F: ...
    def forward(self, arg0: Tensor4F) -> Tensor4F: ...
    def __call__(self, arg0: Tensor4F) -> Tensor4F: ...

class ReLU_f2:
    def __init__(self) -> None: ...
    def backward(self, arg0: MatrixF) -> MatrixF: ...
    def forward(self, arg0: MatrixF) -> MatrixF: ...
    def __call__(self, arg0: MatrixF) -> MatrixF: ...

class ReLU_f3:
    def __init__(self) -> None: ...
    def backward(self, arg0: Tensor3F) -> Tensor3F: ...
    def forward(self, arg0: Tensor3F) -> Tensor3F: ...
    def __call__(self, arg0: Tensor3F) -> Tensor3F: ...

class SGD:
    @overload
    def __init__(self, arg0: float) -> None: ...
    @overload
    def __init__(self, arg0: float, arg1: List[GradHolderF]) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    @overload
    def register_params(self, arg0: List[GradHolderF]) -> None: ...
    @overload
    def register_params(self, arg0: GradHolderF) -> None: ...
    @overload
    def register_params(*args, **kwargs) -> Any: ...
    def step(self) -> None: ...

class Saver:
    def __init__(self, arg0: LayerBase) -> None: ...
    def load(self, arg0: str) -> None: ...
    def save(self, arg0: str) -> None: ...

class Tensor3F(DataHolderF):
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @overload
    def __init__(self, arg0: buffer) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def data_size(self) -> int: ...
    def reshape1(self, *args, **kwargs) -> Any: ...
    def reshape2(self, *args, **kwargs) -> Any: ...
    def reshape3(self, arg0: List[int[3]]) -> Tensor3F: ...
    def reshape4(self, arg0: List[int[4]]) -> Tensor4F: ...
    def shape(self) -> List[int[3]]: ...
    def __getitem__(self, arg0: Tuple[int,int,int]) -> float: ...
    def __setitem__(self, arg0: Tuple[int,int,int], arg1: float) -> None: ...

class Tensor3I(DataHolderI):
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None: ...
    @overload
    def __init__(self, arg0: buffer) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def data_size(self) -> int: ...
    def reshape1(self, *args, **kwargs) -> Any: ...
    def reshape2(self, *args, **kwargs) -> Any: ...
    def reshape3(self, arg0: List[int[3]]) -> Tensor3I: ...
    def reshape4(self, arg0: List[int[4]]) -> Tensor4I: ...
    def shape(self) -> List[int[3]]: ...
    def __getitem__(self, arg0: Tuple[int,int,int]) -> int: ...
    def __setitem__(self, arg0: Tuple[int,int,int], arg1: int) -> None: ...

class Tensor4F(DataHolderF):
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int) -> None: ...
    @overload
    def __init__(self, arg0: buffer) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def data_size(self) -> int: ...
    def reshape1(self, *args, **kwargs) -> Any: ...
    def reshape2(self, *args, **kwargs) -> Any: ...
    def reshape3(self, *args, **kwargs) -> Any: ...
    def reshape4(self, arg0: List[int[4]]) -> Tensor4F: ...
    def shape(self) -> List[int[4]]: ...
    def __getitem__(self, arg0: Tuple[int,int,int,int]) -> float: ...
    def __setitem__(self, arg0: Tuple[int,int,int,int], arg1: float) -> None: ...

class Tensor4I(DataHolderI):
    @overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int) -> None: ...
    @overload
    def __init__(self, arg0: buffer) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def data_size(self) -> int: ...
    def reshape1(self, *args, **kwargs) -> Any: ...
    def reshape2(self, *args, **kwargs) -> Any: ...
    def reshape3(self, *args, **kwargs) -> Any: ...
    def reshape4(self, arg0: List[int[4]]) -> Tensor4I: ...
    def shape(self) -> List[int[4]]: ...
    def __getitem__(self, arg0: Tuple[int,int,int,int]) -> int: ...
    def __setitem__(self, arg0: Tuple[int,int,int,int], arg1: int) -> None: ...

class Variable1F(GradHolderF):
    def __init__(self, arg0: List[int[1]]) -> None: ...
    def grad(self) -> VectorF: ...
    def tensor(self) -> VectorF: ...

class Variable1I(GradHolderI):
    def __init__(self, arg0: List[int[1]]) -> None: ...
    def grad(self) -> VectorI: ...
    def tensor(self) -> VectorI: ...

class Variable2F(GradHolderF):
    def __init__(self, arg0: List[int[2]]) -> None: ...
    def grad(self) -> MatrixF: ...
    def tensor(self) -> MatrixF: ...

class Variable2I(GradHolderI):
    def __init__(self, arg0: List[int[2]]) -> None: ...
    def grad(self) -> MatrixI: ...
    def tensor(self) -> MatrixI: ...

class Variable3F(GradHolderF):
    def __init__(self, arg0: List[int[3]]) -> None: ...
    def grad(self) -> Tensor3F: ...
    def tensor(self) -> Tensor3F: ...

class Variable3I(GradHolderI):
    def __init__(self, arg0: List[int[3]]) -> None: ...
    def grad(self) -> Tensor3I: ...
    def tensor(self) -> Tensor3I: ...

class Variable4F(GradHolderF):
    def __init__(self, arg0: List[int[4]]) -> None: ...
    def grad(self) -> Tensor4F: ...
    def tensor(self) -> Tensor4F: ...

class Variable4I(GradHolderI):
    def __init__(self, arg0: List[int[4]]) -> None: ...
    def grad(self) -> Tensor4I: ...
    def tensor(self) -> Tensor4I: ...

class VectorF(DataHolderF):
    @overload
    def __init__(self, arg0: int) -> None: ...
    @overload
    def __init__(self, arg0: buffer) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def data_size(self) -> int: ...
    def reshape1(self, arg0: List[int[1]]) -> VectorF: ...
    def reshape2(self, arg0: List[int[2]]) -> MatrixF: ...
    def reshape3(self, arg0: List[int[3]]) -> Tensor3F: ...
    def reshape4(self, arg0: List[int[4]]) -> Tensor4F: ...
    def shape(self) -> List[int[1]]: ...
    def __getitem__(self, arg0: int) -> float: ...
    def __setitem__(self, arg0: int, arg1: int) -> None: ...

class VectorI(DataHolderI):
    @overload
    def __init__(self, arg0: int) -> None: ...
    @overload
    def __init__(self, arg0: buffer) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> Any: ...
    def data_size(self) -> int: ...
    def reshape1(self, arg0: List[int[1]]) -> VectorI: ...
    def reshape2(self, arg0: List[int[2]]) -> MatrixI: ...
    def reshape3(self, arg0: List[int[3]]) -> Tensor3I: ...
    def reshape4(self, arg0: List[int[4]]) -> Tensor4I: ...
    def shape(self) -> List[int[1]]: ...
    def __getitem__(self, arg0: int) -> int: ...
    def __setitem__(self, arg0: int, arg1: int) -> None: ...
