#pragma once

#include <tensor/tensor.hpp>

namespace ts {


template <typename Element>
class GradHolder
{
  public:
    using DataHolderRef = DataHolder<Element> &;

    virtual auto grad() -> DataHolderRef  = 0;

    virtual auto weight() -> DataHolderRef = 0;
};


template <typename Element, int Dim>
class Variable : public GradHolder<Element>
{
  public:
    using DataHolderPtr = std::unique_ptr<Tensor<Element, Dim>>;
    using DataHolderRef = Tensor<Element, Dim> &;
    DataHolderPtr _weight;
    DataHolderPtr _grad;

    Variable(DataHolderPtr weight, DataHolderPtr grad)
        : _weight(std::move(weight)), _grad(std::move(grad)) {}

    auto grad() -> DataHolderRef {
        return *_grad;
    }
    auto weight() -> DataHolderRef {
        return *_weight;
    }

    auto set_grad(DataHolderPtr grad) -> void { _grad = std::move(grad);
    }

    auto set_weight(DataHolderPtr weight) -> void {
        _weight = std::move(weight);
    }
};

}
