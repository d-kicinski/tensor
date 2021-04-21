#include <catch2/catch.hpp>
#include <iostream>
#include <tensor/tensor.hpp>
#include <tensor/nn/im2col.hpp>


using namespace ts;

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


    // weight: (m, k) -> (c_out, c_in * k * k)
    // input: (k, n) ->(c_in * k * k, h * w)

    auto const output_shape = compute_output_shape({C, H, W}, K, stride, pad, dilatation);
    ts::Tensor<float, 3> col_out(output_shape);
    im2col_cpu(im_in.raw_data(), C, H, W, K, K, pad, pad, stride, stride, dilatation, dilatation,
               col_out.raw_data_mutable());

    Tensor<float, 3> output(C, H, W);
    col2im_cpu(col_out.raw_data(), C, H, W, K, K, pad, pad, stride, stride, dilatation, dilatation,
               output.raw_data_mutable());
}
