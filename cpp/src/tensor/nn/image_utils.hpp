#pragma once

#include <tensor/tensor.hpp>

namespace ts {

auto pad(MatrixF const &matrix, int pad_row, int pad_col) -> ts::MatrixF;

auto hwc2chw(ts::Tensor<float, 4> const &hwc_tensor) -> ts::Tensor<float, 4>;

auto chw2hwc(ts::Tensor<float, 4> const &chw_tensor) -> ts::Tensor<float, 4>;

} // namespace ts
