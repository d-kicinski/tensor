#pragma once

#include "tensor/tensor.hpp"

namespace ts {

template <typename Element, int Dim> class ActivationBase {
  public:
    auto operator()(Tensor<Element, Dim> const &input) -> Tensor<Element, Dim> { return forward(input); }

    virtual auto backward(Tensor<Element, Dim> const &d_output) -> Tensor<Element, Dim> { return d_output; }

    virtual auto forward(Tensor<Element, Dim> const &input) -> Tensor<Element, Dim> { return input; }
};

template <typename Element, int Dim> class ReLU : public ActivationBase<Element, Dim> {
  public:
    auto backward(Tensor<Element, Dim> const &d_output) -> Tensor<Element, Dim> override
    {
        return ts::assign_if(d_output, _input <= 0, 0.0f); // d_y[_y <= 0] = 0;
    }

    auto forward(Tensor<Element, Dim> const &input) -> Tensor<Element, Dim> override
    {
        _input = input;
        return ts::maximum(0.0f, input); // np.maximum(0, _y)
    }

  private:
    Tensor<Element, Dim> _input;
};

enum class Activation { RELU, NONE };

template <typename Element, int Dim> class ActivationFactory {
  public:
    using ActivationPtr = std::unique_ptr<ActivationBase<Element, Dim>>;
    using OptActivationPtr = std::optional<ActivationPtr>;

    static auto get(Activation activation) -> std::optional<ActivationPtr>
    {
        switch (activation) {
        case Activation::RELU:
            return relu();
        default:
            return std::nullopt;
        }
    }

    static auto relu() -> std::unique_ptr<ReLU<Element, Dim>>
    {
        return std::make_unique<ReLU<Element, Dim>>(ReLU<Element, Dim>());
    }
};

} // namespace ts
