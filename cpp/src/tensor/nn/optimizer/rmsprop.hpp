#pragma once

#include <utility>

#include "tensor/nn/optimizer/optimizer.hpp"
#include "tensor/nn/variable.hpp"
#include "tensor/tensor.hpp"

namespace ts {

template <typename Element> class RMSProp : public Optimizer<Element> {
  public:
    using Ref = std::reference_wrapper<GradHolder<Element>>;
    using VectorRef = std::vector<Ref>;
    using Optimizer<Element>::register_params;

    RMSProp(float lr, float alpha) : _lr(lr), _alpha(alpha) {}

    RMSProp(VectorRef variables, float lr, float alpha) : RMSProp(lr, alpha) { register_params(std::move(variables)); }

    RMSProp(VectorRef variables, float lr) : RMSProp(variables, lr, ALPHA) {}

    explicit RMSProp(VectorRef variables) : RMSProp(variables, LEARNING_RATE, ALPHA) {}

    explicit RMSProp(float lr) : RMSProp(lr, ALPHA) {}

    RMSProp() : RMSProp(LEARNING_RATE, ALPHA) {}

    auto register_params(VectorRef variables) -> void override
    {
        _register_in_memory(variables);
        Optimizer<Element>::register_params(std::move(variables));
    }

    auto step() -> void override
    {
        VectorRef params = Optimizer<Element>::params();
        for (int i = 0; i < _grad_moving_average.size(); ++i) {
            ts::DataHolder<float> &tensor = params[i].get().tensor();
            ts::DataHolder<float> &grad = params[i].get().grad();
            std::vector<Element> &avg = _grad_moving_average[i];

            ts::clip_(grad, -5.0f, 5.0f);

            for (int j = 0; j < avg.size(); ++j) {
                avg[j] = _alpha * avg[j] + (1 - _alpha) * std::pow(grad.at(j), 2);
                tensor.at(j) -= _lr * grad.at(j) / (std::sqrt(avg[j]) + 1e-8);
            }
        }
    }

  private:
    constexpr static double LEARNING_RATE = 1e-2;
    constexpr static double ALPHA = 0.99;

    float _lr{};
    std::vector<std::vector<Element>> _grad_moving_average;
    float _alpha{};

    auto _register_in_memory(VectorRef const &variables) -> void
    {
        for (GradHolder<Element> &item : variables) {
            auto size = std::distance(item.tensor().begin(), item.tensor().end());
            _grad_moving_average.push_back(std::vector<Element>(size));
        }
    }
};

} // namespace ts
