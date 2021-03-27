#pragma once

#include "activations.hpp"
#include "layer_base.hpp"
#include "variable.hpp"
#include <tensor/tensor.hpp>

namespace ts {

class Conv2D : public LayerBase<float> {
  public:
    using Activations = ActivationFactory<float, 4>;
    using VectorRef = std::vector<std::reference_wrapper<GradHolder<float>>>;

    static auto create(int in_channels, int out_channels, int kernel_size, int stride,
                       Activation activation = Activation::NONE, bool use_bias = true) -> Conv2D;

    auto operator()(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto forward(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto backward(Tensor<float, 4> const &) -> Tensor<float, 4>;

    auto weight() -> Variable<float, 2> &;

    auto bias() -> std::optional<std::reference_wrapper<Variable<float, 1>>>;

    auto weights() -> VectorRef;

  private:
    Conv2D(Variable<float, 2> weight, std::optional<Variable<float, 1>> bias, int kernel_size, int stride,
           Activation activation = Activation::NONE);

    Tensor<float, 4> _input;
    Variable<float, 2> _weight;
    std::optional<Variable<float, 1>> _bias;
    Activations::OptActivationPtr _activation;
    int _stride;
    int _kernel_size;
};

} // namespace ts