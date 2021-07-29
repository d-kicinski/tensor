#pragma once

#include <cmath>

#include <tensor/ops_common.hpp>

namespace ts {
template <typename T, int Dim> auto sigmoid(Tensor<T, Dim> const &input) -> Tensor<T, Dim>
{
    constexpr float epsilon = 1e-10;
    Fn<float> sigmoid_f = [](T e) { return T(1) / (T(1) + std::exp(-e) + epsilon); };
    return ts::apply(input, sigmoid_f);
}

template <typename T, int Dim>
auto sigmoid_backward(Tensor<T, Dim> const &output, Tensor<T, Dim> const &d_output) -> Tensor<T, Dim>
{
    Tensor<T, Dim> result(output.shape());
    for (int i = 0; i < output.data_size(); ++i) {
        auto o = output.at(i);
        result.at(i) = o * (1 - o) * d_output.at(i);
    }
    return result;
}

} // namespace ts
