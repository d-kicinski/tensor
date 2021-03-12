#pragma once
#include <tensor/nn/variable.hpp>
#include <tensor/tensor.hpp>
#include <utility>

namespace ts {

template <typename Element> class SGD {
  public:
    using Ref = std::reference_wrapper<GradHolder<Element>>;
    using VectorRef = std::vector<Ref>;
    VectorRef _variables;
    float _lr;
    explicit SGD(float lr) : _lr(lr) {}
    SGD(float lr, VectorRef variables) : _lr(lr), _variables(std::move(variables)) {}

    auto register_params(VectorRef variables_ref) -> void
    {
        _variables.insert(_variables.end(), variables_ref.begin(), variables_ref.end());
    }

    auto register_params(Ref variable_ref) -> void { _variables.push_back(variable_ref); }

    auto step() -> void
    {
        for (Ref &var : _variables) {
            ts::clip_(var.get().grad(), -5.0f, 5.0f);
            // print_stats<Element>(var.get().tensor(), var.get().name() + "[tensor]");
            // print_stats(var.get().grad(), var.get().name() + "[grad]  ");
            std::transform(var.get().tensor().begin(), var.get().tensor().end(),
                           var.get().grad().begin(), var.get().tensor().begin(),
                           [&](Element &w, Element &d_w) { return w - (_lr * d_w); });
        }
    }
};

} // namespace ts
