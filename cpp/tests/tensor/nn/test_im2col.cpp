#include <catch2/catch.hpp>
#include <iostream>
#include <tensor/tensor.hpp>
#include <tensor/nn/im2col.hpp>


using namespace ts;


auto compute_output_shape(std::array<int, 3> const &input_shape, int kernel_size, int stride,
                          int pad, int dilatation) -> std::array<int, 3>
{
    // col_buffer_shape: c_in * k * k, out_1, out_2
    std::array<int, 3> output_shape{input_shape[0] * kernel_size * kernel_size};

    int const num_spatial_axes_ = 2;

    for (int i = 0; i < num_spatial_axes_; ++i) {
        int const input_dim = input_shape[i + 1];
        int const kernel_extent = dilatation * (kernel_size - 1) + 1;
        int const output_dim = (input_dim + 2 * pad - kernel_extent) / stride + 1;
        output_shape[i+1] = output_dim;
    }
    return output_shape;
}
TEST_CASE("im2col")
{
    int C = 3;
    int H = 4;
    int W = 4;
    int K = 3;
    int stride = 1;
    int pad = 1;
    int dilatation = 1;

    Tensor<float, 3> im_in(C, H, W);
    std::iota(im_in.begin(), im_in.end(), 1);

    auto const output_shape = compute_output_shape({C, H, W}, K, stride, pad, dilatation);
    ts::Tensor<float, 3> col_out(output_shape);
    im2col_cpu(im_in.raw_data(), C, H, W, K, K, pad, pad, stride, stride, dilatation, dilatation,
               col_out.raw_data_mutable());

    Tensor<float, 3> output(C, H, W);
    col2im_cpu(col_out.raw_data(), C, H, W, K, K, pad, pad, stride, stride, dilatation, dilatation,
               output.raw_data_mutable());
}
