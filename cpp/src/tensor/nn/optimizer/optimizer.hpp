#pragma once

#include "tensor/nn/parameters_registry.hpp"
#include "tensor/ops_common.hpp"

namespace ts {

template <typename T> class Optimizer : public ParameterRegistry<T> {
  public:
    auto zero_gradients() -> void
    {
        for (auto &item : ParameterRegistry<T>::parameters()) {
            ts::fill_(item.get().grad(), T(0));
        }
    }

    virtual auto step() -> void = 0;
};

} // namespace ts
