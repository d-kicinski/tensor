#pragma once

#include <tensor/tensor.hpp>
#include <tensor/ops.hpp>

namespace ts {

auto softmax(MatrixF const &logits) -> MatrixF
{
    return ts::divide(ts::exp(logits), ts::sum(ts::exp(logits), 1));
}

}