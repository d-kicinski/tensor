#pragma once

#include <utility>

#include "tensor/nn/optimizer/optimizer.hpp"
#include "tensor/nn/variable.hpp"
#include "tensor/tensor.hpp"

namespace ts {

template <typename Element> class SGD : public Optimizer<Element> {
  public:
    using Ref = std::reference_wrapper<GradHolder<Element>>;
    using VectorRef = std::vector<Ref>;
    using Optimizer<Element>::register_params;

    SGD(float lr, float momentum) : _lr(lr), _momentum(momentum)
    {
        if (_momentum > 0.0) {
            _previous_updates = std::make_optional(std::vector<std::vector<Element>>());
        }
    }

    SGD(VectorRef variables, float lr, float momentum) : SGD(lr, momentum) { register_params(variables); }

    SGD(VectorRef variables, float lr) : SGD(variables, lr, MOMENTUM) {}

    explicit SGD(VectorRef variables) : SGD(variables, LEARNING_RATE, MOMENTUM) {}

    explicit SGD(float lr) : SGD(lr, MOMENTUM) {}

    SGD() : SGD(LEARNING_RATE, MOMENTUM) {}

    auto register_params(VectorRef variables) -> void override
    {
        if (_momentum > 0) {
            _update_momentum_buffer(variables);
        }
        Optimizer<Element>::register_params(std::move(variables));
    }

    auto step() -> void
    {
        auto params = Optimizer<Element>::params();
        for (int i = 0; i < params.size(); ++i) {
            DataHolder<Element> &tensor = params[i].get().tensor();
            DataHolder<Element> &grad = params[i].get().grad();

            ts::clip_(grad, -5.0f, 5.0f);

            if (_previous_updates) {
                std::vector<Element> &prev = _previous_updates.value()[i];
                for (int j = 0; j < prev.size(); ++j) {
                    prev[j] = _momentum * prev[j] + _lr * grad.at(j);
                    tensor.at(j) += -prev[j];
                }
            } else {
                std::transform(tensor.begin(), tensor.end(), grad.begin(), tensor.begin(),
                               [&](Element &w, Element &d_w) { return w - (_lr * d_w); });
            }
        }
    }

  private:
    constexpr static double LEARNING_RATE = 1e-3;
    constexpr static double MOMENTUM = 0.0;

    float _lr{};
    float const _momentum{};
    std::optional<std::vector<std::vector<Element>>> _previous_updates = std::nullopt;

    auto _update_momentum_buffer(VectorRef variables) -> void
    {
        assert(_previous_updates.has_value());
        for (GradHolder<Element> &item : variables) {
            auto size = std::distance(item.tensor().begin(), item.tensor().end());
            _previous_updates.value().push_back(std::vector<Element>(size));
        }
    }
};

} // namespace ts
