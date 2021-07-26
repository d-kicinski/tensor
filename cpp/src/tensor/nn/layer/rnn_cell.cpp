#include "rnn_cell.hpp"

ts::RNNCell::RNNCell(ts::RNNCell::Parameters &p, int vocab_size) : _p(p), _vocab_size(vocab_size) {}

auto ts::RNNCell::forward(int input_index, const ts::MatrixF &previous_hidden_state) -> ts::MatrixF
{
    _input = MatrixF(1, _vocab_size);
    _input.at({0, input_index}) = 1;
    _previous_hidden_state = previous_hidden_state.clone();

    auto x1 = ts::dot(_input, _p.wxh.tensor());
    auto x2 = ts::dot(_previous_hidden_state, _p.whh.tensor());
    auto x3 = ts::add(x2,  _p.bh.tensor());
    auto x4 = ts::add(x1, x3);
    _hidden_state = ts::tanh(x4);

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

    MatrixF d_tanh(d_hidden.shape());
    for (int i = 0; i < d_tanh.data_size(); ++i) {
        d_tanh.at(i) = (1 - _hidden_state.at(i) * _hidden_state.at(i)) * d_hidden.at(i);
    }
    //    auto d_tanh = ts::tanh_backward(_hidden_state, d_hidden);

    _p.bh.grad() += ts::sum(d_tanh, 0);
    // [vocab_size, hidden_size] = [batch_size, vocab_size].T x [batch_size, hidden_size]
    _p.wxh.grad() += ts::dot(_input, d_tanh, true, false);
    // [hidden_size, hidden_size] = [batch_size, hidden_size].T x [batch_size, hidden_size]
    _p.whh.grad() += ts::dot(_previous_hidden_state, d_tanh, true, false); // [hidden_size, hidden_size]
    // [batch_size, hidden_size] = [batch_size, hidden_size] x [hidden_size, hidden_size].T
    return ts::dot(d_tanh, _p.whh.tensor(), false, true);
}
auto ts::RNNCell::hidden_state() -> ts::MatrixF & { return _hidden_state; }

auto ts::RNNCell::loss(int input_index, const ts::MatrixF &previous_hidden_state) -> MatrixF
{
    auto output = forward(input_index, previous_hidden_state);
//    VectorI targets{target_index};

    auto exp_logits = ts::exp(output);
    auto sum_exp = ts::sum(exp_logits, 1);
    _probs = ts::divide(exp_logits, sum_exp);
    return _probs;
//    return _loss_fn.forward(output, targets);
}
auto ts::RNNCell::loss_backward(int target_index, ts::MatrixF const &next_d_hidden_state) -> ts::MatrixF
{
//    auto d_scores = _loss_fn.backward();
    auto d_scores = _probs.clone();
    d_scores.at({0, target_index}) -= 1;
    return backward(d_scores, next_d_hidden_state);
}
