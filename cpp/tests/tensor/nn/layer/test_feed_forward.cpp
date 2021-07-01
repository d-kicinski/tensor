#include <catch2/catch.hpp>

#include <tensor/nn/layer/feed_forward.hpp>

using namespace ts;

TEST_CASE("Create FeedForward layer")
{
    auto layer = FeedForward::create(2, 100);
    REQUIRE(true);
}

TEST_CASE("FeedForward: forward, backward")
{
    auto layer = FeedForward::create(2, 3);
    MatrixF input(32, 2);
    auto y = layer(input);
    {
        std::array<size_type, 2> expected_shape = {32, 3};
        REQUIRE(y.shape() == expected_shape);
    }
    auto d_y = layer.backward(y);
    {
        std::array<size_type, 2> expected_shape = {32, 2};
        REQUIRE(d_y.shape() == expected_shape);
    }
}
