#include "softmax.hpp"

using namespace ts;

auto ts::softmax(MatrixF const &logits) -> MatrixF
{
    return ts::divide(ts::exp(logits), ts::sum(ts::exp(logits), 1));
}
