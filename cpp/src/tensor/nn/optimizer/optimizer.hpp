#pragma once

#include <utility>

#include "tensor/nn/variable.hpp"

namespace ts {

template <typename Element> class Optimizer {
  public:
    using Ref = std::reference_wrapper<GradHolder<Element>>;
    using VectorRef = std::vector<Ref>;

    virtual auto register_params(VectorRef variables_ref) -> void
    {
        _variables.insert(_variables.end(), variables_ref.begin(), variables_ref.end());
    }

    virtual auto register_params(Ref variable_ref) -> void { _variables.push_back(variable_ref); }

    auto params() -> VectorRef { return _variables; }

    virtual auto step() -> void = 0;

  private:
    VectorRef _variables;
};

} // namespace ts
