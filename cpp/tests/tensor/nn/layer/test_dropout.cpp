#include <catch2/catch.hpp>

#include <tensor/nn/layer/dropout.hpp>

TEST_CASE("dropout")
{
    float keep_probability = 0.5;
    auto dropout = ts::Dropout(keep_probability);

    ts::MatrixF input {
        {10, 10, 10},
        {20, 20, 20}
    };
    auto result = dropout.forward(input);

    ts::MatrixF expected {
        {20, 0, 20},
        {0, 40, 0}
    };
    REQUIRE(result == expected);

}
