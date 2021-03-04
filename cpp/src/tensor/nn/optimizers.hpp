#pragma once
#include <tensor/tensor.hpp>
#include <utility>
#include <tensor/nn/variable.hpp>

namespace ts {

template<typename Element>
class SGD
{
  public:
    using VectorRef = std::vector<std::reference_wrapper<GradHolder<float>>>;
    VectorRef _variables;
    float _lr;
    SGD(float lr, VectorRef variables) : _lr(lr), _variables(std::move(variables)) {}

    auto step() -> void {
        for (auto var : _variables) {
            std::transform(var.get().weight().begin(), var.get().weight().end(), var.get().grad().begin(), var.get().weight().begin(),
                           [&](Element & w, Element & d_w) {
                             return w + (d_w * -_lr);
                           });
        }
    }
};

}
