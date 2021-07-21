#pragma once

#include <utility>

#include "tensor/nn/optimizer/optimizer.hpp"
#include "tensor/nn/variable.hpp"
#include "tensor/tensor.hpp"

namespace ts {

template <typename Element> class Adam : public Optimizer<Element> {
  public:
    using Ref = std::reference_wrapper<GradHolder<Element>>;
    using VectorRef = std::vector<Ref>;

    Adam(VectorRef variables, float lr, float beta1, float beta2) : _lr(lr), _beta1(beta1), _beta2(beta2)
    {
        _register_in_memory(variables);
        Optimizer<Element>::register_params(std::move(variables));
    }

    Adam(float lr, float beta1, float beta2) : _lr(lr), _beta1(beta1), _beta2(beta2) {}

    Adam(VectorRef variables, float lr) : Adam(variables, lr, BETA1, BETA2) {}

    explicit Adam(VectorRef variables) : Adam(variables, LEARNING_RATE, BETA1, BETA2) {}

    explicit Adam(float lr) : Adam(lr, BETA1, BETA2) {}

    Adam() : Adam(LEARNING_RATE, BETA1, BETA2) {}

    auto register_params(VectorRef variables) -> void override
    {
        _register_in_memory(variables);
        Optimizer<Element>::register_params(variables);
    }
    auto register_params(Ref variables) -> void override
    {
        _register_in_memory(std::vector<Ref>{variables});
        Optimizer<Element>::register_params(variables);
    }

    auto step() -> void override
    {
        // It's a bit different than what's in paper or on wiki but PyTorch implementation does this, idk
        float bias_correction1 = 1 - std::pow(_beta1, _step);
        float bias_correction2 = 1 - std::pow(_beta1, _step);

        VectorRef params = Optimizer<Element>::params();
        for (int i = 0; i < _grad_moving_average.size(); ++i) {
            ts::DataHolder<float> &tensor = params[i].get().tensor();
            ts::DataHolder<float> &grad = params[i].get().grad();
            std::vector<Element> &grad_avg = _grad_moving_average[i];
            std::vector<Element> &grad2_avg = _grad_squared_moving_average[i];

            ts::clip_(grad, -5.0f, 5.0f);

            for (int j = 0; j < grad_avg.size(); ++j) {
                grad_avg[j] = _beta1 * grad_avg[j] + (1 - _beta1) * grad.at(j);
                grad2_avg[j] = _beta2 * grad2_avg[j] + (1 - _beta2) * std::pow(grad.at(j), 2);
                float denom = std::sqrt(grad2_avg[j]) / (std::sqrt(bias_correction2) + 1e-8);
                float step_size = _lr / bias_correction1;
                tensor.at(j) -= step_size * grad_avg[j] / denom;
            }
        }
        _step++;
    }

  private:
    constexpr static double LEARNING_RATE = 1e-3;
    constexpr static double BETA1 = 0.9;
    constexpr static double BETA2 = 0.999;

    float _lr{};
    std::vector<std::vector<Element>> _grad_moving_average;
    std::vector<std::vector<Element>> _grad_squared_moving_average;
    float _beta1{};
    float _beta2{};
    int _step = 1;

    auto _register_in_memory(VectorRef variables) -> void
    {
        for (GradHolder<Element> &item : variables) {
            auto size = std::distance(item.tensor().begin(), item.tensor().end());
            _grad_moving_average.push_back(std::vector<Element>(size));
            _grad_squared_moving_average.push_back(std::vector<Element>(size));
        }
    }
};

} // namespace ts
