#include <catch2/catch.hpp>

#include <tensor/nn/conv2d.hpp>
#include <tensor/nn/functional_helpers.hpp>

using namespace ts;

TEST_CASE("Conv2d(..., use_bias = false")
{
    size_type batch_size = 16;
    size_type kernel_size = 3;
    size_type channel_in = 3;
    size_type channel_out = 9;
    size_type dim_in = 64;
    size_type dim_out = _calculate_output_dim(dim_in, kernel_size, 0, 1, 1);

    auto layer = Conv2D::create(channel_in, channel_out, kernel_size, 1, Activation::NONE, false);
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
