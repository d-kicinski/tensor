#pragma once

#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

namespace ts {

auto softmax(MatrixF const &logits) -> MatrixF;

auto log_softmax(MatrixF const &logits) -> MatrixF;

} // namespace ts