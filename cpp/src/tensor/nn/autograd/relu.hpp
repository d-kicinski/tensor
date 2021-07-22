#pragma once

#include <tensor/ops_common.hpp>

namespace ts {
template <typename Element, int Dim> auto relu(Tensor<Element, Dim> const &input) -> Tensor<Element, Dim>
{
    return ts::maximum(0.0f, input); // np.maximum(0, _y)
}

template <typename Element, int Dim>
auto relu_backward(Tensor<Element, Dim> &input, Tensor<Element, Dim> const &d_output) -> Tensor<Element, Dim>
{
    return ts::assign_if(d_output, input <= 0, 0.0f); // d_y[_y <= 0] = 0;
}

} // namespace ts