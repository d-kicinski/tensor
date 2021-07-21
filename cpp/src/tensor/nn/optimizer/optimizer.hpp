#pragma once

#include <utility>

#include "tensor/nn/variable.hpp"

namespace ts {

template <typename Element> class Optimizer {
  public:
    using Ref = std::reference_wrapper<GradHolder<Element>>;
    using VectorRef = std::vector<Ref>;

    auto params() -> VectorRef { return _variables; }

    auto register_params(Ref variable_ref) -> void { register_params(std::vector<Ref>{variable_ref}); }

    virtual auto register_params(VectorRef variables_ref) -> void
    {
        _variables.insert(_variables.end(), variables_ref.begin(), variables_ref.end());
    }

    virtual auto step() -> void = 0;

  private:
    VectorRef _variables;
};

} // namespace ts
