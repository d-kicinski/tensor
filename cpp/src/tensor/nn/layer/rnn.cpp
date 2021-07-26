#include "rnn.hpp"

ts::RNN::RNN(int hidden_size, int sequence_length, int vocab_size)
    : _hidden_size(hidden_size), _sequence_length(sequence_length), _vocab_size(vocab_size),
      _p{Variable<float, 2>::create(ts::standard_normal<float, 2>({vocab_size, hidden_size}, 0.01)),
         Variable<float, 2>::create(ts::standard_normal<float, 2>({hidden_size, hidden_size}, 0.01)),
         Variable<float, 2>::create(ts::standard_normal<float, 2>({hidden_size, vocab_size}, 0.01)),
         Variable<float, 1>::create(ts::Tensor<float, 1>(hidden_size)),
         Variable<float, 1>::create(ts::Tensor<float, 1>(vocab_size))}
{
    for (int i = 0; i < sequence_length; ++i) {
        _cells.emplace_back(_p, vocab_size);
    }
    register_parameters(_p.wxh);
    register_parameters(_p.whh);
    register_parameters(_p.why);
    register_parameters(_p.bh);
    register_parameters(_p.by);
}
auto ts::RNN::forward(std::vector<int> inputs, std::vector<int> targets, ts::MatrixF const &previous_state) -> float
{
    _last_state = previous_state; // not a deep copy but we won't be modifying content so its fine
    float loss = 0.0;
    for (int i = 0; i < inputs.size(); ++i) {
        auto &cell = _cells[i];
        auto output = cell.forward(inputs[i], _last_state);
        loss += cell.loss().forward(output, {targets[i]});
        _last_state = cell.hidden_state(); // deep copy doesn't make much sense here
    }
    return loss;
}
auto ts::RNN::backward() -> void
{
    auto next_d_hidden_state = MatrixF(1, _hidden_size);
    for (int i = _sequence_length - 1; i >= 0; --i) {
        auto &cell = _cells[i];
        auto d_scores = cell.loss().backward();
        next_d_hidden_state = cell.backward(d_scores, next_d_hidden_state);
    }
}
auto ts::RNN::state() -> ts::MatrixF & { return _last_state; }

auto ts::RNN::sample(int idx, ts::MatrixF const &previous_state, int sample_size) -> std::vector<int>
{
    auto cell = RNNCell(_p, _vocab_size);
    auto prev_state = previous_state;
    std::vector<int> indices(sample_size);
    indices[0] = idx;

    std::discrete_distribution<> distribution;
    std::default_random_engine random;

    for (int i = 0; i < sample_size - 1; ++i) {
        auto probabilities = ts::softmax(cell.forward(indices[i], prev_state));
        distribution.param({probabilities.begin(), probabilities.end()});
        indices[i + 1] = distribution(random);
        prev_state = cell.hidden_state();
    }
    return indices;
}
