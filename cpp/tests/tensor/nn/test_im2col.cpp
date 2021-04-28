#include <catch2/catch.hpp>
#include <tensor/tensor.hpp>
#include <tensor/nn/im2col.hpp>


using namespace ts;

TEST_CASE("im2col")
{
    size_type C = 3;
    size_type H = 4;
    size_type W = 4;
    int kernel = 3;
    int stride = 1;
    int pad = 1;
    int dilatation = 1;

    Tensor<float, 3> im_in(C, H, W);
    std::iota(im_in.begin(), im_in.end(), 1);


    // weight: (m, k) -> (c_out, c_in * k * k)
    // input: (k, n) ->(c_in * k * k, h * w)
    auto const output_shape = ts::im2col::im2col_buffer_shape({C, H, W}, kernel, stride, pad, dilatation);
    ts::Tensor<float, 2> col_out(output_shape);
    ts::im2col::im2col(im_in, kernel, pad, stride, dilatation, col_out);

    Tensor<float, 3> output(C, H, W);
    ts::im2col::col2im(col_out, kernel, pad, stride, dilatation, output);
}
