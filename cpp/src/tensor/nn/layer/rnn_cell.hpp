#pragma once

#include "tensor/nn/cross_entropy_loss.hpp"
#include "tensor/nn/layer/feed_forward.hpp"
#include "tensor/nn/parameters_registry.hpp"
#include "tensor/nn/softmax.hpp"

namespace ts {

class RNNCell {
  public:
    struct Parameters {
        Variable<float, 2> wxh;
        Variable<float, 2> whh;
        Variable<float, 2> why;
        Variable<float, 1> bh;
        Variable<float, 1> by;
    };

    RNNCell(Parameters &p, int vocab_size);

    auto forward(int input_index, MatrixF const &previous_hidden_state) -> MatrixF;

    auto backward(MatrixF const &d_output, MatrixF const &next_d_hidden_state) -> MatrixF;

    auto hidden_state() -> MatrixF &;

    auto loss(int input_index, MatrixF const &previous_hidden_state) -> MatrixF;

    auto loss_backward(int target_index, MatrixF const &next_d_hidden_state) -> MatrixF;

  private:
    Parameters &_p;
    int _vocab_size;
    MatrixF _hidden_state;
    CrossEntropyLoss _loss_fn;
    MatrixF _input;
    MatrixF _previous_hidden_state;
    MatrixF _probs;
};
} // namespace ts
