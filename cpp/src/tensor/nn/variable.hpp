#pragma once

#include "grad_holder.hpp"
#include <tensor/tensor.hpp>
#include <utility>

namespace ts {

template <typename Element, int Dim> class Variable : public GradHolder<Element> {
  public:
    using DataHolderPtr = std::unique_ptr<Tensor<Element, Dim>>;
    using DataHolderRef = Tensor<Element, Dim> &;

    template <typename... Sizes> static auto create(Sizes... args) -> Variable
    {
        auto weight = std::make_unique<Tensor<Element, Dim>>(Tensor<Element, Dim>(args...));
        auto grad = std::make_unique<Tensor<Element, Dim>>(Tensor<Element, Dim>(weight->shape()));
        return Variable(std::move(weight), std::move(grad), "Variable");
    }

    explicit Variable(std::array<size_type, Dim> const &shape)
    {
        _weight = std::make_unique<Tensor<Element, Dim>>(Tensor<Element, Dim>(shape));
        _grad = std::make_unique<Tensor<Element, Dim>>(Tensor<Element, Dim>(_weight->shape()));
    }

    Variable(DataHolderPtr &&weight, DataHolderPtr &&grad) : Variable(weight, grad, "Variable") {}

    Variable(DataHolderPtr &&weight, DataHolderPtr &&grad, std::string name)
        : _weight(std::move(weight)), _grad(std::move(grad)), _name(std::move(name))
    {
    }

    auto grad() -> DataHolderRef override { return *_grad; }
    auto tensor() -> DataHolderRef override { return *_weight; }
    auto name() -> std::string override { return _name; };
    auto set_grad(DataHolderPtr grad) -> void { _grad = std::move(grad); }
    auto set_weight(DataHolderPtr weight) -> void { _weight = std::move(weight); }

  private:
    DataHolderPtr _weight;
    DataHolderPtr _grad;
    std::string _name;
};

} // namespace ts
