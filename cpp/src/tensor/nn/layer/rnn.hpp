#pragma once

#include "feed_forward.hpp"
#include "rnn_cell.hpp"

namespace ts {
class RNN : public LayerBase<float> {
  public:
    RNN(int hidden_size, int sequence_length, int vocab_size)
        : _hidden_size(hidden_size), _sequence_length(sequence_length), _vocab_size(vocab_size),
          _input2hidden(vocab_size, hidden_size), _hidden2hidden(hidden_size, hidden_size),
          _hidden2output(hidden_size, vocab_size),
          _initial_state(1, hidden_size)
    {
        for (int i = 0; i < sequence_length; ++i) {
            _cells.emplace_back(_input2hidden, _hidden2hidden, _hidden2output, vocab_size);
        }
        register_parameters(_input2hidden.parameters());
        register_parameters(_hidden2hidden.parameters());
        register_parameters(_hidden2output.parameters());
    }

    auto forward(std::vector<int> inputs, std::vector<int> targets, MatrixF const &previous_state) -> float
    {
        _initial_state = previous_state.clone();
        _last_state = previous_state.clone();
        float loss = 0.0;
        for (int i = 0; i < inputs.size(); ++i) {
            auto &cell = _cells[i];
            loss += cell.loss(inputs[i], targets[i], _last_state);
            _last_state = cell.hidden_state().clone();
        }
        return loss;
    }

    auto backward() -> void
    {
        auto next_d_hidden_state = MatrixF(1, _hidden_size);
        auto previous_hidden_state = _initial_state.clone();
        for (int i = 0; i < _sequence_length; ++i) {
            auto &cell = _cells[i];
            next_d_hidden_state = cell.loss_backward(previous_hidden_state, next_d_hidden_state);
            previous_hidden_state = cell.hidden_state().clone();
        }
    }

    auto sample(int idx, MatrixF previous_state, int sample_size) -> std::vector<int>
    {
        auto cell = RNNCell(_input2hidden, _hidden2hidden, _hidden2output, _vocab_size);
        auto &prev_state = previous_state;
        std::vector<int> indices(sample_size);
        indices[0] = idx;
        for (int i = 0; i < sample_size - 1; ++i) {
            indices[i + 1] = ts::argmax(cell.forward(indices[i], prev_state)).at(0);
            prev_state = cell.hidden_state();
        }
        return indices;
    }

    auto state() -> MatrixF& {return _last_state;}

  private:
    int _hidden_size;
    int _sequence_length;
    int _vocab_size;
    FeedForward _input2hidden;
    FeedForward _hidden2hidden;
    FeedForward _hidden2output;
    std::vector<RNNCell> _cells;
    MatrixF _initial_state;
    MatrixF _last_state;
};
} // namespace ts