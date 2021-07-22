#pragma once

#include <cmath>

#include <tensor/ops_common.hpp>

namespace ts {
template <typename Element, int Dim> auto tanh(Tensor<Element, Dim> const &input) -> Tensor<Element, Dim>
{
    constexpr float epsilon = 1e-10;
    Fn<float> tanh = [](Element e) { return std::tanh(e + epsilon); };
    return ts::apply(input, tanh);
}

template <typename Element, int Dim>
auto tanh_backward(Tensor<Element, Dim> const &output, Tensor<Element, Dim> const &d_output) -> Tensor<Element, Dim>
{
    Tensor<Element, Dim> result(output.shape());
    for (int i = 0; i < output.data_size(); ++i) {
        result.at(i) = (1 - std::pow(output.at(i), 2)) * d_output.at(i);
    }
    return result;
}

} // namespace ts