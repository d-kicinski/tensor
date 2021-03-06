#pragma once

#include <tensor/tensor.hpp>

namespace ts {


template <typename Element>
class GradHolder
{
  public:
    using DataHolderRef = DataHolder<Element> &;

    DataHolder<Element> _grad;
    DataHolder<Element> _weight;

    GradHolder() : _grad(DataHolder<Element>()), _weight(DataHolder<Element>()) {}

    virtual auto grad() -> DataHolderRef  {
        return _grad;
    };

    virtual auto tensor() -> DataHolderRef {
        return _weight;
    };
};


template <typename Element, int Dim>
class Variable : public GradHolder<Element>
{
  public:
    using DataHolderPtr = std::unique_ptr<Tensor<Element, Dim>>;
    using DataHolderRef = Tensor<Element, Dim> &;
    DataHolderPtr _weight;
    DataHolderPtr _grad;

    template <typename... Sizes>
    static auto create(Sizes... args) -> Variable {
        auto weight = std::make_unique<Tensor<Element, Dim>>(Tensor<Element, Dim>(args...));
        auto grad = std::make_unique<Tensor<Element, Dim>>(Tensor<Element, Dim>(weight->shape()));
        return Variable(std::move(weight), std::move(grad));
    }

    Variable(): _weight(nullptr), _grad(nullptr) {}

    explicit Variable(std::array<int , Dim> const & shape) {
        _weight = std::make_unique<Tensor<Element, Dim>>(Tensor<Element, Dim>(shape));
        _grad = std::make_unique<Tensor<Element, Dim>>(Tensor<Element, Dim>(_weight->shape()));
    }

    Variable(DataHolderPtr && weight, DataHolderPtr && grad)
        : _weight(std::move(weight)), _grad(std::move(grad)) {}

    auto grad() -> DataHolderRef {
        return *_grad;
    }
    auto tensor() -> DataHolderRef {
        return *_weight;
    }

    auto set_grad(DataHolderPtr grad) -> void { _grad = std::move(grad);
    }

    auto set_weight(DataHolderPtr weight) -> void {
        _weight = std::move(weight);
    }
};

}
