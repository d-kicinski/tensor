#pragma once

#include <tensor/tensor.hpp>

namespace ts {

template <typename T> class LayerBase {
  private:
    using params_t = std::vector<std::reference_wrapper<DataHolder<T>>>;
    params_t _params;

  public:
    auto register_parameter(DataHolder<T> &param) -> void { _params.push_back(std::ref(param)); }

    auto register_parameters(params_t &params) -> void
    {
        for (auto param : params) {
            _params.push_back(param);
        }
    }

    auto parameters() -> params_t & { return _params; }
};

} // namespace ts
