#pragma once

#include <tensor/tensor.hpp>
#include <tensor/nn/activations.hpp>

namespace ts {

class Conv2D {
  public:
    using OptActivation = std::optional<Activation<float, 3>>;
    using Activations = ActivationFactory<float, 3>;

    Conv2D(int in_channels, int out_channels, int kernel_size, int stride,
           OptActivation activation = std::nullopt,
           bool use_bias = true);

    auto operator()(Tensor<float, 3> const &) -> Tensor<float, 3>;

    auto forward(Tensor<float, 3> const &) -> Tensor<float, 3>;

    auto backward(Tensor<float, 3> const &) -> Tensor<float, 3>;

    auto update(float step_size) -> void;

    auto weight() -> MatrixF;

    auto bias() -> VectorF;

  private:
    Tensor<float, 3> _input;
    MatrixF _weight;
    MatrixF _d_weight;
    VectorF _bias;
    VectorF _d_bias;
    OptActivation _activation;
    int _stride;
    int _kernel_size;
    bool _use_bias;
};

} // namespace ts