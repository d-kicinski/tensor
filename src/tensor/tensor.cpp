#include "tensor.hpp"

namespace ts {

template class Tensor<float, 1>;
template class Tensor<float, 2>;
template class Tensor<float, 3>;

template class Tensor<int, 1>;
template class Tensor<int, 2>;
template class Tensor<int, 3>;

}