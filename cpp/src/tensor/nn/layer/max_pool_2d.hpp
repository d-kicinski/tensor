#pragma once

#include <tensor/tensor.hpp>

#include "tensor/nn/variable.hpp"

namespace ts {

class MaxPool2D {
  public:
    static auto create(int kernel_size, int stride, int pad) -> MaxPool2D;

    MaxPool2D(int kernel_size, int stride, int pad);

    auto operator()(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto forward(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto backward(Tensor<float, 4> const &) -> Tensor<float, 4>;

  private:
    Tensor<int, 4> _mask;
    int _dim_in{};
    int _kernel_size;
    int _stride;
    int _pad;
};

} // namespace ts
