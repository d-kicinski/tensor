#include <catch2/catch.hpp>

#include <tensor/nn/feedforward.hpp>

using namespace ts::nn;

TEST_CASE("Create FeedForward layer")
{
    FeedForward layer(2, 100);
    REQUIRE(true);
}


TEST_CASE("ff: forward, backward")
{
    FeedForward layer(2, 3);
    ts::Matrix input(32, 2);
    auto y = layer(input);
    {
        std::vector<int> expected_shape = {32, 3};
        REQUIRE(y.shape() == expected_shape);
    }
    auto d_y =layer.backward(y);
    {
        std::vector<int> expected_shape = {32, 2};
        REQUIRE(d_y.shape() == expected_shape);
    }
    layer.update(0.001);
}
