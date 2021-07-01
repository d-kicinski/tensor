#include <catch2/catch.hpp>

#include <tensor/nn/conv_2d_helpers.hpp>
#include <tensor/nn/layer/conv_2d_im2col.hpp>
#include <tensor/nn/layer/conv_2d_naive.hpp>

using namespace ts;

TEST_CASE("conv2d_naive(..., use_bias = false")
{
    size_type batch_size = 16;
    size_type kernel_size = 3;
    size_type channel_in = 3;
    size_type channel_out = 9;
    size_type dim_in = 64;
    size_type dim_out = _calculate_output_dim(dim_in, kernel_size, 0, 1, 1);

    auto layer = naive::Conv2D::create(channel_in, channel_out, kernel_size, 1, Activation::NONE, false);
    Tensor<float, 4> input(batch_size, dim_in, dim_in, channel_in);

    auto output = layer(input);

    {
        std::array<size_type, 4> expected_shape = {batch_size, dim_out, dim_out, channel_out};
        REQUIRE(output.shape() == expected_shape);
    }
    auto d_input = layer.backward(output);
    {
        std::array<size_type , 4> expected_shape = {batch_size, dim_in, dim_in, channel_in};
        REQUIRE(d_input.shape() == expected_shape);
    }

}

TEST_CASE("conv2d_im2col(..., use_bias = false")
{
    size_type batch_size = 16;
    size_type kernel_size = 3;
    size_type channel_in = 3;
    size_type channel_out = 9;
    size_type dim_in = 64;
    size_type dim_out = _calculate_output_dim(dim_in, kernel_size, 0, 1, 1);

    auto layer = im2col::Conv2D::create(channel_in, channel_out, kernel_size, 1, 0, 1, Activation::NONE, false);
    Tensor<float, 4> input(batch_size, channel_in, dim_in, dim_in);

    auto output = layer(input);

    {
        std::array<size_type, 4> expected_shape = {batch_size, channel_out, dim_out, dim_out};
        REQUIRE(output.shape() == expected_shape);
    }
    auto d_input = layer.backward(output);
    {
        std::array<size_type , 4> expected_shape = {batch_size, channel_in, dim_in, dim_in};
        REQUIRE(d_input.shape() == expected_shape);
    }

}
