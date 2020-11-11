#pragma once

#include <tensor/tensor.hpp>

namespace ts::nn {

class FeedForward {
  public:
    FeedForward(int dim_in, int dim_out, float alpha=1e-3, bool activation=false);

    auto operator()(Matrix) -> Matrix;

    auto forward(Matrix) -> Matrix;

    auto backward(Matrix) -> Matrix;

    auto update(int step_size) -> void;

  private:
    Matrix _x;
    Matrix _y;
    Matrix _weights;
    Matrix _d_weights;
    Matrix _bias;
    Matrix _d_bias;
    float _alpha;
    bool _activation;
};

}

