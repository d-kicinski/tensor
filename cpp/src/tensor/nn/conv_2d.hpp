#pragma once

#include <tensor/tensor_forward.hpp>
#include <tuple>

namespace ts {

auto conv_2d_im2col(ts::Tensor<float, 4> const &images, ts::Tensor<float, 2> const &kernel,
                    ts::Tensor<float, 2> &im2col_buffer, int kernel_size, int stride, int pad, int dilatation)
    -> ts::Tensor<float, 4>;

auto conv_2d(ts::Tensor<float, 4> const &images, ts::Tensor<float, 2> const &kernel, int kernel_size, size_type stride)
    -> ts::Tensor<float, 4>;

auto conv_2d(Tensor<float, 3> const &image, Tensor<float, 2> const &kernel, int kernel_size, size_type stride)
    -> Tensor<float, 3>;

auto conv_2d(MatrixF const &matrix, MatrixF const &kernel, size_type stride) -> MatrixF;

auto conv_2d_backward(Tensor<float, 4> const &input, Tensor<float, 2> const &kernel, Tensor<float, 4> const &d_output,
                      int kernel_size, int stride) -> std::tuple<Tensor<float, 4>, Tensor<float, 2>>;

auto conv_2d_backward_im2col(ts::Tensor<float, 4> const &inputs, ts::Tensor<float, 2> const &kernel,
                             ts::Tensor<float, 2> &im2col_buffer, ts::Tensor<float, 4> const &d_outputs,
                             int kernel_size, int stride, int pad, int dilatation)
    -> std::tuple<ts::Tensor<float, 4>, ts::Tensor<float, 2>>;

auto conv_2d_backward(Tensor<float, 3> const &input, Tensor<float, 2> const &kernel, Tensor<float, 3> const &d_output,
                      int kernel_size, int stride) -> std::tuple<Tensor<float, 3>, Tensor<float, 2>>;

} // namespace ts
