#include "cross_entropy_loss.hpp"
#include "softmax.hpp"

auto ts::CrossEntropyLoss::operator()(const ts::MatrixF &probs, ts::Tensor<int, 1> const &labels) -> float
{
    return forward(probs, labels);
}

auto ts::CrossEntropyLoss::forward(const ts::MatrixF &logits, ts::Tensor<int, 1> const &labels) -> float
{
    _labels = labels;
    _scores = ts::softmax(logits);
    auto log_probs = -ts::log(ts::get(_scores, labels));
    float loss = ts::sum(log_probs) / _scores.shape(0);
    return loss;
}

auto ts::CrossEntropyLoss::backward() -> ts::MatrixF
{
    auto d_scores = ts::apply_if(
        _scores, ts::to_one_hot(_labels), (Fn<float>)[](float e) { return e - 1; });
    d_scores = ts::apply(
        d_scores, (Fn<float>)[&](float e) { return e / _scores.shape(0); });
    return d_scores;
}
