#pragma once

#include "tensor/nn/grad_holder.hpp"

namespace ts {

template <typename T> class LayerBase {
  public:
    using VectorRef = std::vector<std::reference_wrapper<GradHolder<T>>>;

    auto register_parameters(GradHolder<T> &param) -> void { _params.push_back(std::ref(param)); }

    auto register_parameters(VectorRef &params) -> void
    {
        for (auto param : params) {
            _params.push_back(param);
        }
    }

    auto parameters() -> VectorRef & { return _params; }

  private:
    VectorRef _params;

};

} // namespace ts
