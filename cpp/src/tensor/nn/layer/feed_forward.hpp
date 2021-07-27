#pragma once

#include <optional>

#include "tensor/nn/activations.hpp"
#include "tensor/nn/parameters_registry.hpp"
#include "tensor/nn/variable.hpp"
#include "tensor/tensor.hpp"

namespace ts {

class FeedForward : public ParameterRegistry<float> {
  public:
    using Activations = ActivationFactory<float, 2>;
    using VectorRef = std::vector<std::reference_wrapper<GradHolder<float>>>;

    FeedForward(int dim_in, int dim_out, Activation activation = Activation::NONE, bool use_bias = true);

    static auto create(int dim_in, int dim_out, Activation activation = Activation::NONE, bool use_bias = true)
        -> FeedForward;

    auto operator()(MatrixF const &) -> MatrixF;

    auto forward(MatrixF const &) -> MatrixF;

    auto backward(MatrixF const &) -> MatrixF;

    auto weight() -> Variable<float, 2> &;

    auto bias() -> std::optional<std::reference_wrapper<Variable<float, 1>>>;

    auto weights() -> VectorRef;

  private:
    FeedForward(Variable<float, 2> weight, std::optional<Variable<float, 1>> bias,
                Activation activation = Activation::NONE);

    Variable<float, 2> _weight;
    std::optional<Variable<float, 1>> _bias = std::nullopt;
    Activations::OptActivationPtr _activation;
    bool _use_bias;

    MatrixF _x{};
};

} // namespace ts
