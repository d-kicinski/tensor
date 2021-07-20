#pragma once

#include <utility>

#include "tensor/nn/optimizer/optimizer.hpp"
#include "tensor/nn/variable.hpp"
#include "tensor/tensor.hpp"

namespace ts {

template <typename Element> class Adagrad : public Optimizer<Element> {
  public:
    using Ref = std::reference_wrapper<GradHolder<Element>>;
    using VectorRef = std::vector<Ref>;

    float _lr;
    std::vector<std::vector<Element>> _memory;

    explicit Adagrad(float lr) : _lr(lr) {}

    Adagrad(float lr, VectorRef variables) : _lr(lr)
    {
        _register_in_memory(variables);
        Optimizer<Element>::register_params(std::move(variables));
    }

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
        VectorRef params = Optimizer<Element>::params();
        for (int i = 0; i < _memory.size(); ++i) {
            DataHolder<Element> &tensor = params[i].get().tensor();
            DataHolder<Element> &grad = params[i].get().grad();
            std::vector<Element> &mem = _memory[i];

            ts::clip_(grad, -5.0f, 5.0f);

            for (int j = 0; j < mem.size(); ++j) {
                mem[j] += grad.begin()[j] * grad.begin()[j];
                tensor.begin()[j] += -_lr * grad.begin()[j] / std::sqrt(mem[j] + 1e-8);
            }
        }
    }

  private:
    auto _register_in_memory(VectorRef variables) -> void
    {
        for (GradHolder<Element> &item : variables) {
            auto size = std::distance(item.tensor().begin(), item.tensor().end());
            _memory.push_back(std::vector<Element>(size));
        }
    }
};

} // namespace ts