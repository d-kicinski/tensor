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

    explicit SGD(float lr, float momentum = 0.0) : _lr(lr), _momentum(momentum)
    {
        if (_momentum > 0.0) {
            _previous_updates = std::make_optional(std::vector<std::vector<Element>>());
        }
    }

    SGD(float lr, VectorRef variables, float momentum = 0.0) : _lr(lr), _momentum(momentum)
    {
        if (_momentum > 0.0) {
            _previous_updates = std::make_optional(std::vector<std::vector<Element>>());
            _update_momentum_buffer(variables);
        }
        Optimizer<Element>::register_params(std::move(variables));
    }

    auto register_params(VectorRef variables) -> void override
    {
        if (_momentum > 0) {
            _update_momentum_buffer(variables);
        }
        Optimizer<Element>::register_params(variables);
    }
    auto register_params(Ref variables) -> void override
    {
        if (_momentum > 0) {
            _update_momentum_buffer(std::vector<Ref>{variables});
        }
        Optimizer<Element>::register_params(variables);
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
                    prev[j] = _momentum * prev[j] + _lr * grad.begin()[j];
                    tensor.begin()[j] += -prev[j];
                }
            } else {
                std::transform(tensor.begin(), tensor.end(), grad.begin(), tensor.begin(),
                               [&](Element &w, Element &d_w) { return w - (_lr * d_w); });
            }
        }
    }

  private:
    float _lr;
    float const _momentum;
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
