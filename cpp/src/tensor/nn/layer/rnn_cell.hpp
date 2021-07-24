#pragma once

#include "tensor/nn/layer/feed_forward.hpp"
#include "tensor/nn/layer/layer_base.hpp"
#include <tensor/nn/cross_entropy_loss.hpp>
#include <tensor/nn/softmax.hpp>

namespace ts {

class RNNCell {
  public:
    RNNCell(FeedForward &input2hidden, FeedForward &hidden2hidden, FeedForward &hidden2output, int vocab_size)
        : _input2hidden(input2hidden), _hidden2hidden(hidden2hidden), _hidden2output(hidden2output),
          _vocab_size(vocab_size)
    {
    }

    auto forward(int input_index, MatrixF const &previous_hidden_state) -> MatrixF
    {
        auto input = MatrixF(_vocab_size, 1);
        input.at(input_index) = 1;
        _hidden_state = ts::tanh(ts::add(_input2hidden(input), _hidden2hidden(previous_hidden_state)));
        auto output = _hidden2output(_hidden_state);
        return output;
    }

    auto backward(MatrixF const &d_output, MatrixF const &previous_hidden_state, MatrixF const &next_d_hidden_state)
        -> MatrixF
    {
        auto d_hidden = _hidden2output.backward(d_output);
        auto d_tanh = ts::tanh_backward(_hidden_state, d_hidden);
        auto d_input = _input2hidden.backward(d_tanh);
        auto d_previous_hidden = _hidden2hidden.backward(d_tanh);
        return d_previous_hidden;
    }

    auto hidden_state() -> MatrixF & { return _hidden_state; }

    auto loss(int input_index, int target_index, MatrixF const &previous_hidden_state) -> float
    {
        auto output = forward(input_index, previous_hidden_state);
        auto targets = VectorI{target_index};
        return _loss_fn.forward(output, targets);
    }

    auto loss_backward(MatrixF const &previous_hidden_state, MatrixF const &next_d_hidden_state)
    {
        auto d_scores = _loss_fn.backward();
        return backward(d_scores, previous_hidden_state, next_d_hidden_state);
    }

  private:
    FeedForward &_input2hidden;
    FeedForward &_hidden2hidden;
    FeedForward &_hidden2output;
    int _vocab_size;
    MatrixF _hidden_state;
    CrossEntropyLoss _loss_fn;
};
} // namespace ts
