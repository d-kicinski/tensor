#pragma once

#include "activations.hpp"
#include <tensor/tensor.hpp>

namespace ts {

class FeedForward {
  public:
    using OptActivation = std::optional<std::unique_ptr<Activation<float, 2>>>;
    using Activations = ActivationFactory<float, 2>;

    FeedForward(int dim_in, int dim_out, OptActivation activation = std::nullopt, bool l2 = false,
                float alpha = 1e-10);

    auto operator()(MatrixF const &) -> MatrixF;

    auto forward(MatrixF const &) -> MatrixF;

    auto backward(MatrixF const &) -> MatrixF;

    auto update(float step_size) -> void;

    auto weight() -> MatrixF;

    auto bias() -> VectorF;

  private:
    MatrixF _x;
    MatrixF _weights;
    MatrixF _d_weights;
    VectorF _bias;
    VectorF _d_bias;
    OptActivation _activation;
    float _alpha;
    bool _l2;
};

} // namespace ts
