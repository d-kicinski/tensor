#include "lstm.hpp"

#include <utility>

#include "tensor/nn/initialization.hpp"
#include "tensor/nn/softmax.hpp"

ts::LSTM::LSTM(int vocab_size, int sequence_length, int hidden_size, int embedding_dim)
    : _hidden_size(hidden_size), _sequence_length(sequence_length), _vocab_size(vocab_size),
      _concat_size(embedding_dim + hidden_size),
      _p{Variable<float, 2>::create(ts::uniform<float, 2>({_concat_size, hidden_size}, hidden_size)),
         Variable<float, 2>::create(ts::uniform<float, 2>({_concat_size, hidden_size}, hidden_size)),
         Variable<float, 2>::create(ts::uniform<float, 2>({_concat_size, hidden_size}, hidden_size)),
         Variable<float, 2>::create(ts::uniform<float, 2>({_concat_size, hidden_size}, hidden_size)),
         Variable<float, 1>::create(ts::uniform<float, 1>({hidden_size}, hidden_size)),
         Variable<float, 1>::create(ts::uniform<float, 1>({hidden_size}, hidden_size)),
         Variable<float, 1>::create(ts::uniform<float, 1>({hidden_size}, hidden_size)),
         Variable<float, 1>::create(ts::uniform<float, 1>({hidden_size}, hidden_size))},
      _index2hidden(vocab_size, hidden_size, Activation::NONE, false),
      _hidden2output(hidden_size, vocab_size, Activation::NONE, false), _last_memory(1, hidden_size),
      _last_state(1, hidden_size)

{
    for (int i = 0; i < sequence_length; ++i) {
        _cells.emplace_back(_p);
    }
    register_parameters(_p.wxc);
    register_parameters(_p.wxf);
    register_parameters(_p.wxi);
    register_parameters(_p.wxo);
    register_parameters(_p.bc);
    register_parameters(_p.bf);
    register_parameters(_p.bi);
    register_parameters(_p.bo);
    register_parameters(_index2hidden.parameters());
    register_parameters(_hidden2output.parameters());
}

auto ts::LSTM::forward(std::vector<int> inputs, std::vector<int> targets, ts::MatrixF const &previous_state,
                       ts::MatrixF const &previous_memory) -> float
{
    _last_state = previous_state;
    _last_memory = previous_memory;
    float loss = 0.0;
    for (int i = 0; i < inputs.size(); ++i) {
        MatrixF input(1, _vocab_size);
        input.at({0, inputs[i]}) = 1;
        _input_embedding = _index2hidden.forward(input);
        auto &cell = _cells[i];
        auto hidden_state = cell.forward(_input_embedding, _last_state, _last_memory);
        auto logits = _hidden2output.forward(hidden_state);
        loss += cell.loss().forward(logits, {targets[i]});

        _last_state = cell.state();
        _last_memory = cell.memory();
    }
    return loss;
}

auto ts::LSTM::forward(std::vector<int> inputs, std::vector<int> targets) -> float
{
    return forward(std::move(inputs), std::move(targets), _last_state, _last_memory);
}

auto ts::LSTM::backward() -> void
{
    auto d_hidden_state = MatrixF(1, _hidden_size);
    auto d_memory_state = MatrixF(1, _hidden_size);

    for (int i = _sequence_length - 1; i >= 0; --i) {
        auto &cell = _cells[i];
        auto d_scores = cell.loss().backward();
        auto d_hidden = _hidden2output.backward(d_scores);
        d_hidden_state += d_hidden;
        auto [d_h, d_c, d_x] = cell.backward(d_hidden_state, d_memory_state);
        _index2hidden.backward(d_x);
        d_hidden_state = d_h;
        d_memory_state = d_c;
    }
}

auto ts::LSTM::sample(int idx, ts::MatrixF const &previous_state, ts::MatrixF const &previous_memory, int sample_size)
    -> std::vector<int>
{
    auto cell = ts::LSTMCell(_p);
    auto prev_h = previous_state;
    auto prev_c = previous_memory;

    std::vector<int> indices(sample_size);
    indices[0] = idx;

    std::discrete_distribution<> distribution;
    std::default_random_engine random;

    for (int i = 0; i < sample_size - 1; ++i) {
        MatrixF input(1, _vocab_size);
        input.at({0, indices[i]}) = 1;
        auto probabilities = ts::softmax(_hidden2output(cell.forward(_index2hidden(input), prev_h, prev_c)));

        distribution.param({probabilities.begin(), probabilities.end()});
        indices[i + 1] = distribution(random);
        prev_h = cell.state();
        prev_c = cell.memory();
    }
    return indices;
}

auto ts::LSTM::last_state() -> ts::MatrixF & { return _last_state; }

auto ts::LSTM::last_memory() -> ts::MatrixF & { return _last_memory; }
