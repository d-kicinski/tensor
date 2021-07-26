#pragma once

#include <utility>

#include "tensor/nn/optimizer/optimizer.hpp"
#include "tensor/nn/variable.hpp"
#include "tensor/tensor.hpp"

namespace ts {

template <typename T> class Adagrad : public Optimizer<T> {
  public:
    using Optimizer<T>::register_parameters;
    using VectorRef = typename Optimizer<T>::VectorRef;

    explicit Adagrad(float lr) : _lr(lr) {}

    Adagrad(VectorRef variables, float lr) : Adagrad(lr) { register_parameters(std::move(variables)); }

    explicit Adagrad(VectorRef variables) : Adagrad(variables, LEARNING_RATE) {}

    Adagrad() : Adagrad(LEARNING_RATE) {}

    auto register_parameters(VectorRef variables) -> void override
    {
        _register_in_memory(variables);
        Optimizer<T>::register_parameters(std::move(variables));
    }

    auto step() -> void override
    {
        VectorRef params = Optimizer<T>::parameters();
        for (int i = 0; i < _memory.size(); ++i) {
            DataHolder<T> &tensor = params[i].get().tensor();
            DataHolder<T> &grad = params[i].get().grad();
            std::vector<T> &mem = _memory[i];

            ts::clip_(grad, -5.0f, 5.0f);

            for (int j = 0; j < mem.size(); ++j) {
                mem[j] += grad.at(j) * grad.at(j);
                tensor.at(j) += -_lr * grad.at(j) / std::sqrt(mem[j] + 1e-8);
            }
        }
    }

  private:
    constexpr static double LEARNING_RATE = 1e-2;

    float _lr{};
    std::vector<std::vector<T>> _memory;

    auto _register_in_memory(VectorRef variables) -> void
    {
        for (GradHolder<T> &item : variables) {
            auto size = std::distance(item.tensor().begin(), item.tensor().end());
            _memory.push_back(std::vector<T>(size));
        }
    }
};

} // namespace ts
