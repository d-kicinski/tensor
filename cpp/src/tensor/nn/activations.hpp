#pragma once

#include "tensor/tensor.hpp"
#include <tensor/nn/autograd/relu.hpp>
#include <tensor/nn/autograd/tanh.hpp>

namespace ts {

template <typename Element, int Dim> class ActivationBase {
  public:
    auto operator()(Tensor<Element, Dim> const &input) -> Tensor<Element, Dim> { return forward(input); }

    virtual auto forward(Tensor<Element, Dim> const &input) -> Tensor<Element, Dim> { return input; }

    virtual auto backward(Tensor<Element, Dim> const &d_output) -> Tensor<Element, Dim> { return d_output; }
};

template <typename Element, int Dim> class ReLU : public ActivationBase<Element, Dim> {
  public:
    auto forward(Tensor<Element, Dim> const &input) -> Tensor<Element, Dim> override
    {
        _input = input;
        return ts::relu(_input);
    }

    auto backward(Tensor<Element, Dim> const &d_output) -> Tensor<Element, Dim> override
    {
        return ts::relu_backward<Element, Dim>(_input, d_output);
    }

  private:
    Tensor<Element, Dim> _input;
};

template <typename Element, int Dim> class Tanh : public ActivationBase<Element, Dim> {
  public:
    auto forward(Tensor<Element, Dim> const &input) -> Tensor<Element, Dim> override
    {
        _output = ts::tanh(input);
        return _output;
    }

    auto backward(Tensor<Element, Dim> const &d_output) -> Tensor<Element, Dim> override
    {
        return ts::tanh_backward<Element, Dim>(_output, d_output);
    }

  private:
    Tensor<Element, Dim> _output;
};

enum class Activation { RELU, TANH, NONE };

template <typename Element, int Dim> class ActivationFactory {
  public:
    using ActivationPtr = std::unique_ptr<ActivationBase<Element, Dim>>;
    using OptActivationPtr = std::optional<ActivationPtr>;

    static auto get(Activation activation) -> std::optional<ActivationPtr>
    {
        switch (activation) {
        case Activation::RELU:
            return std::make_unique<ReLU<Element, Dim>>(ReLU<Element, Dim>());
        case Activation::TANH:
            return std::make_unique<Tanh<Element, Dim>>(Tanh<Element, Dim>());
        default:
            return std::nullopt;
        }
    }
};

} // namespace ts
