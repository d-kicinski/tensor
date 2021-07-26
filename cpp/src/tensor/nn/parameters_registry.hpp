#pragma once

#include "tensor/nn/grad_holder.hpp"

namespace ts {

template <typename T> class ParameterRegistry {
  public:
    using Ref = std::reference_wrapper<GradHolder<T>>;
    using VectorRef = std::vector<Ref>;

    virtual auto register_parameters(GradHolder<T> &param) -> void
    {
        register_parameters(std::vector<Ref>{std::ref(param)});
    }

    virtual auto register_parameters(VectorRef params) -> void
    {
        for (const auto &item : params) {
            _params.push_back(item);
        }
    }

    auto parameters() -> VectorRef & { return _params; }

  private:
    VectorRef _params;
};

} // namespace ts
