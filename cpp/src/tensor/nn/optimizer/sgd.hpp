#pragma once

#include <utility>

#include "tensor/nn/optimizer/optimizer.hpp"
#include "tensor/nn/variable.hpp"
#include "tensor/tensor.hpp"

namespace ts {

template <typename Element> class SGD : public Optimizer<Element> {
  public:
    using Ref = std::reference_wrapper<GradHolder<Element>>;
    using VectorRef = std::vector<Ref>;

    float _lr;

    explicit SGD(float lr) : _lr(lr) {}
    SGD(float lr, VectorRef variables) : _lr(lr) {
        Optimizer<Element>::register_params(std::move(variables));
    }

    auto step() -> void
    {
        for (Ref &var : Optimizer<Element>::params()) {
            ts::clip_(var.get().grad(), -5.0f, 5.0f);
            std::transform(var.get().tensor().begin(), var.get().tensor().end(), var.get().grad().begin(),
                           var.get().tensor().begin(), [&](Element &w, Element &d_w) { return w - (_lr * d_w); });
        }
    }
};

} // namespace ts
