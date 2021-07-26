#include "rnn_cell.hpp"

ts::RNNCell::RNNCell(ts::RNNCell::Parameters &p, int vocab_size) : _p(p), _vocab_size(vocab_size) {}

auto ts::RNNCell::forward(int input_index, ts::MatrixF const &previous_hidden_state) -> ts::MatrixF
{
    _input = MatrixF(1, _vocab_size);
    _input.at({0, input_index}) = 1;
    _previous_hidden_state = previous_hidden_state; // not a deep copy, we trust that underlying data won't we changed

    auto input_contrib = ts::dot(_input, _p.wxh.tensor());
    auto prev_h_contrib = ts::add(ts::dot(_previous_hidden_state, _p.whh.tensor()), _p.bh.tensor());
    _hidden_state = ts::tanh(ts::add(input_contrib, prev_h_contrib));

    auto output = ts::add(ts::dot(_hidden_state, _p.why.tensor()), _p.by.tensor());
    return output;
}
auto ts::RNNCell::backward(ts::MatrixF const &d_output, ts::MatrixF const &next_d_hidden_state) -> ts::MatrixF
{
    // [hidden_size, vocab_size] += [batch_size, hidden_size].T x [batch_size, vocab_size]
    _p.why.grad() += ts::dot(_hidden_state, d_output, true, false); // [hidden_size, vocab_size]
    _p.by.grad() += ts::sum(d_output, 0);

    // [batch_size, hidden_size] = [batch_size, vocab_size] x [hidden_size, vocab_size].T
    auto d_hidden = ts::add(ts::dot(d_output, _p.why.tensor(), false, true), next_d_hidden_state);

    auto d_tanh = ts::tanh_backward(_hidden_state, d_hidden);

    _p.bh.grad() += ts::sum(d_tanh, 0);
    // [vocab_size, hidden_size] = [batch_size, vocab_size].T x [batch_size, hidden_size]
    _p.wxh.grad() += ts::dot(_input, d_tanh, true, false);
    // [hidden_size, hidden_size] = [batch_size, hidden_size].T x [batch_size, hidden_size]
    _p.whh.grad() += ts::dot(_previous_hidden_state, d_tanh, true, false); // [hidden_size, hidden_size]
    // [batch_size, hidden_size] = [batch_size, hidden_size] x [hidden_size, hidden_size].T
    return ts::dot(d_tanh, _p.whh.tensor(), false, true);
}
auto ts::RNNCell::hidden_state() -> ts::MatrixF & { return _hidden_state; }

auto ts::RNNCell::loss() -> CrossEntropyLoss & { return _loss_fn; }
