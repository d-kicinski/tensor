#pragma once

#include <tensor/tensor.hpp>
#include "variable.hpp"

namespace ts {

class MaxPool2D {
  public:

    static auto create(int kernel_size, int stride) -> MaxPool2D;

    MaxPool2D(int kernel_size, int stride);

    auto operator()(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto forward(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto backward(Tensor<float, 4> const &) -> Tensor<float, 4>;

  private:
    Tensor<bool, 4> _mask;
    int _kernel_size;
    int _stride;
};

} // namespace ts
