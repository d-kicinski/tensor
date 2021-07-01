#pragma once

#include <tensor/tensor_forward.hpp>

namespace ts {

auto max_pool_2d(ts::Tensor<float, 4> const &input, int kernel_size, int stride, int pad)
    -> std::pair<ts::Tensor<float, 4>, ts::Tensor<int, 4>>;

auto max_pool_2d_backward(ts::Tensor<float, 4> const &d_output, ts::Tensor<int, 4> const &mask, int dim_in,
                          int kernel_size, int stride) -> ts::Tensor<float, 4>;

auto max_pool_2d_hwc(ts::Tensor<float, 4> const &input, int kernel_size, int stride)
    -> std::pair<ts::Tensor<float, 4>, ts::Tensor<bool, 4>>;

auto max_pool_2d_backward_hwc(ts::Tensor<float, 4> const &d_output, ts::Tensor<bool, 4> const &mask, int kernel_size,
                              int stride) -> ts::Tensor<float, 4>;

} // namespace ts
