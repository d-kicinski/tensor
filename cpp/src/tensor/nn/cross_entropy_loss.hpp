#pragma once

#include <tensor/tensor.hpp>
#include <utility>

namespace ts {

class CrossEntropyLoss {

  public:
    auto operator()(Matrix const & probs, Tensor<int, 1> const &labels) -> float;

    auto forward(Matrix const & probs, Tensor<int, 1> const &labels) -> float;

    auto backward() -> Matrix;

  private:
    Tensor<int, 1> _labels;
    Matrix _scores;
};

}

