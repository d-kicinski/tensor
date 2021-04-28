#pragma once

#include <array>
#include <tensor/tensor_forward.hpp>

namespace ts::im2col {

auto im2col_buffer_shape(std::array<size_type, 3> const &input_shape, int kernel_size, int stride, int pad,
                         int dilatation) -> std::array<ts::size_type, 2>;

void im2col(ts::Tensor<float, 3> &image, int kernel, int pad, int stride, int dilation, ts::Tensor<float, 2> &buffer);

void col2im(ts::Tensor<float, 2> &buffer, int kernel, int pad, int stride, int dilation, ts::Tensor<float, 3> &image);

} // namespace ts::im2col
