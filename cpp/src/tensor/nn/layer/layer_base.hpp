#pragma once

#include "tensor/nn/grad_holder.hpp"

namespace ts {

template <typename T> class LayerBase {
  private:
    using params_t = std::vector<std::reference_wrapper<GradHolder<T>>>;
    params_t _params;

  public:
    auto register_parameters(GradHolder<T> &param) -> void { _params.push_back(std::ref(param)); }

    auto register_parameters(params_t &params) -> void
    {
        for (auto param : params) {
            _params.push_back(param);
        }
    }

    auto parameters() -> params_t & { return _params; }
};

} // namespace ts
