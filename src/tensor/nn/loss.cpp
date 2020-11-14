#include "loss.hpp"

auto ts::Loss::operator()(const ts::Matrix &probs, ts::Tensor<int, 1> const & labels) -> float
{
    return forward(probs, labels);
}

auto ts::Loss::forward(const ts::Matrix &probs, ts::Tensor<int, 1> const & labels) -> float
{
    _labels = labels;
    auto log_probs = -ts::log(ts::get(probs, labels));

    // compute the loss: average cross-entropy loss and regularization
}
