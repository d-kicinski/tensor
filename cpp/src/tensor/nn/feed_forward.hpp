#pragma once

#include <tensor/tensor.hpp>
#include "activations.hpp"
#include "variable.hpp"

namespace ts {

class FeedForward {
  public:
    using Activations = ActivationFactory<float, 2>;
    using VectorRef = std::vector<std::reference_wrapper<GradHolder<float>>>;

    static auto create(int dim_in, int dim_out, Activation activation = Activation::NONE, bool l2 = false,
                float alpha = 1e-10) -> FeedForward;

    auto operator()(MatrixF const &) -> MatrixF;

    auto forward(MatrixF const &) -> MatrixF;

    auto backward(MatrixF const &) -> MatrixF;

    auto weight() -> Variable<float, 2> &;

    auto bias() -> Variable<float, 1> &;

    auto weights() -> VectorRef;

  private:
    FeedForward(Variable<float, 2> weight,
                Variable<float, 1> bias,
                Activation activation = Activation::NONE,
                bool l2 = false, float alpha = 1e-10);

    MatrixF _x;
    Variable<float, 2> _weight;
    Variable<float, 1> _bias;
    Activations::OptActivationPtr _activation;
    float _alpha;
    bool _l2;
};

} // namespace ts
