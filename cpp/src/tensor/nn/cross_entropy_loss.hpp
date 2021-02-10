#pragma once

#include <tensor/tensor.hpp>
#include <utility>

namespace ts {

class CrossEntropyLoss {

  public:
    auto operator()(MatrixF const & probs, Tensor<int, 1> const &labels) -> float;

    auto forward(MatrixF const & probs, Tensor<int, 1> const &labels) -> float;

    auto backward() -> MatrixF;

  private:
    Tensor<int, 1> _labels;
    MatrixF _scores;
};

}

