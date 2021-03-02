#pragma once

#include <tensor/tensor.hpp>
#include <tensor/nn/activations.hpp>

namespace ts {

class Conv2D {
  public:
    using Activations = ActivationFactory<float, 4>;

    Conv2D(int in_channels, int out_channels, int kernel_size, int stride,
           Activation activation = Activation::NONE,
           bool use_bias = true);

    auto operator()(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto forward(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto backward(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto update(float step_size) -> void;

    auto weight() -> MatrixF;

    auto bias() -> VectorF;

  private:
    Tensor<float, 4> _input;
    MatrixF _weight;
    MatrixF _d_weight;
    VectorF _bias;
    VectorF _d_bias;
    Activations::OptActivationPtr _activation;
    int _stride;
    int _kernel_size;
    bool _use_bias;
};

} // namespace ts