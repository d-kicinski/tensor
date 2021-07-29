#pragma once

#include "tensor/nn/layer/feed_forward.hpp"
#include "tensor/nn/layer/lstm_cell.hpp"
#include "tensor/nn/parameters_registry.hpp"

namespace ts {

class LSTM : public ParameterRegistry<float> {
  public:
    LSTM(int vocab_size, int sequence_length, int hidden_size, int embedding_dim);

    auto forward(std::vector<int> inputs, std::vector<int> targets) -> float;

    auto forward(std::vector<int> inputs, std::vector<int> targets, MatrixF const &previous_state,
                 MatrixF const &previous_memory) -> float;

    auto backward() -> void;

    auto sample(int idx, ts::MatrixF const &previous_state, ts::MatrixF const &previous_memory, int sample_size)
        -> std::vector<int>;

    auto last_state() -> MatrixF &;

    auto last_memory() -> MatrixF &;

  private:
    int _hidden_size;
    int _sequence_length;
    int _vocab_size;
    int _concat_size;
    LSTMCell::Parameters _p;
    FeedForward _index2hidden;
    FeedForward _hidden2output;
    MatrixF _last_memory;
    MatrixF _last_state;

    std::vector<LSTMCell> _cells{};
    MatrixF _input_embedding{};
};

} // namespace ts