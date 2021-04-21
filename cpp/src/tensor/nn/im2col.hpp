#pragma once

#include <array>
#include <tensor/tensor_forward.hpp>

namespace ts {

auto im2col_buffer_shape(std::array<int, 3> const &input_shape, int kernel_size, int stride, int pad, int dilatation)
    -> std::array<ts::size_type, 3>
{
    // col_buffer_shape: c_in * k * k, out_1, out_2
    std::array<size_type, 3> output_shape{static_cast<size_type>(input_shape[0] * kernel_size * kernel_size)};

    int const num_spatial_axes_ = 2;

    for (int i = 0; i < num_spatial_axes_; ++i) {
        int const input_dim = input_shape[i + 1];
        int const kernel_extent = dilatation * (kernel_size - 1) + 1;
        int const output_dim = (input_dim + 2 * pad - kernel_extent) / stride + 1;
        output_shape[i + 1] = output_dim;
    }
    return output_shape;
}

template <typename Dtype>
void im2col(const Dtype *data_im, int channels, int height, int width, int kernel_h, int kernel_w, int pad_h,
                int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, Dtype *data_col);

template <typename Dtype>
void col2im(const Dtype *data_col, int channels, int height, int width, int kernel_h, int kernel_w, int pad_h,
                int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, Dtype *data_im);
} // namespace ts
