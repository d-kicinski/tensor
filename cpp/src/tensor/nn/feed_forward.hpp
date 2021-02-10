#pragma once

#include <tensor/tensor.hpp>

namespace ts {

class FeedForward {
  public:
    FeedForward(int dim_in, int dim_out, bool activation=false, bool l2=false, float alpha=1e-10);

    auto operator()(MatrixF const &) -> MatrixF;

    auto forward(MatrixF const &) -> MatrixF;

    auto backward(MatrixF) -> MatrixF;

    auto update(float step_size) -> void;

    auto weights() -> MatrixF;

  private:
    MatrixF _x;
    MatrixF _y;
    MatrixF _weights;
    MatrixF _d_weights;
    VectorF _bias;
    VectorF _d_bias;
    float _alpha;
    bool _activation;
    bool _l2;

};

}

