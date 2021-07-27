#pragma once

#include "tensor/nn/parameters_registry.hpp"
#include "tensor/nn/layer/rnn_cell.hpp"

namespace ts {

class RNN : public ParameterRegistry<float> {
  public:
    RNN(int hidden_size, int sequence_length, int vocab_size);

    auto forward(std::vector<int> inputs, std::vector<int> targets, MatrixF const &previous_state) -> float;

    auto backward() -> void;

    auto sample(int idx, MatrixF const &previous_state, int sample_size) -> std::vector<int>;

    auto state() -> MatrixF &;

  private:
    int _hidden_size;
    int _sequence_length;
    int _vocab_size;
    RNNCell::Parameters _p;

    std::vector<RNNCell> _cells{};
    MatrixF _last_state{};
};

} // namespace ts