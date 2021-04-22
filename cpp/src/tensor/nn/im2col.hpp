#pragma once

#include <array>
#include <tensor/tensor_forward.hpp>

namespace ts::im2col {

auto im2col_buffer_shape(std::array<size_type, 3> const &input_shape, int kernel_size, int stride, int pad,
                         int dilatation) -> std::array<ts::size_type, 2>;

template <typename Dtype>
void im2col(const Dtype *data_im, int channels, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w,
            int stride_h, int stride_w, int dilation_h, int dilation_w, Dtype *data_col);

template <typename Dtype>
void col2im(const Dtype *data_col, int channels, int height, int width, int kernel_h, int kernel_w, int pad_h,
            int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, Dtype *data_im);
} // namespace ts::im2col
