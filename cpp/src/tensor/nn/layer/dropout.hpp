#pragma once

#include "tensor/nn/initialization.hpp"
#include "tensor/tensor.hpp"

namespace ts {

class Dropout {
  public:
    explicit Dropout(float keep_probability);

    auto operator()(MatrixF const &input) -> MatrixF;

    auto forward(MatrixF const &input) -> MatrixF;

    auto backward(MatrixF const &d_output) -> MatrixF;

  private:
    float _p;

    MatrixF _weight{};
};
} // namespace ts