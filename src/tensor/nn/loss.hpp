#pragma once

#include <tensor/tensor.hpp>

namespace ts {

class Loss {
  public:
    Loss(std::vector<Matrix> weights, int batch_size, float alpha=1e-3);

    auto operator()(Matrix const & probs, Vector labels) -> float;

    auto forward(Matrix const & probs, Vector labels) -> float;

    auto backward(Matrix const & scores) -> Matrix;

  private:
    float _alpha;
    int _batch_size;
    std::vector<Matrix> _weights;

};

}

