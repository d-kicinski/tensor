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

    VectorF log_probs = ts::log(ts::get(_scores, labels));
    return -ts::sum(log_probs) / static_cast<float>(log_probs.shape(0));
}

auto ts::CrossEntropyLoss::backward() -> ts::MatrixF
{
    auto d_scores = ts::apply_if(
        _scores, ts::to_one_hot(_labels, _scores.shape(1)), (Fn<float>)[](float e) { return e - 1.0f; });
    if (size_type batch_size = _scores.shape(0); batch_size > 1) {
        d_scores = ts::apply(
            d_scores, (Fn<float>)[&](float e) { return e / static_cast<float>(batch_size); });
    }
    return d_scores;
}
