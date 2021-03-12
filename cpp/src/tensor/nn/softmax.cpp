#include "softmax.hpp"

using namespace ts;

auto ts::softmax(MatrixF const &logits) -> MatrixF
{
    constexpr float epsilon = 1e-7f;
    constexpr float almost_one = 1.0f - epsilon;

    auto logits_clone = logits.clone();
    float c = *std::max_element(logits_clone.begin(), logits_clone.end());
    subtract_(logits_clone, c);

    auto exp_logits = ts::exp(logits_clone);
    auto sum_exp = ts::sum(exp_logits, 1);
    auto probs = ts::divide(exp_logits, sum_exp);
    ts::clip_(probs, epsilon, almost_one);
    return probs;
}

auto ts::log_softmax(MatrixF const &logits) -> MatrixF
{
    auto logits_clone = logits.clone();
    float c = *std::max_element(logits_clone.begin(), logits_clone.end());
    subtract_(logits_clone, c);

    auto exp_logits = exp(logits_clone);
    auto sum_exp_logits = sum(exp_logits, 1);
    auto log_sum_exp_logits = log(sum_exp_logits);
    auto neg_log_sum_exp_logits = -log_sum_exp_logits;
    MatrixF result =  add(logits_clone, neg_log_sum_exp_logits);
    return result;
}
