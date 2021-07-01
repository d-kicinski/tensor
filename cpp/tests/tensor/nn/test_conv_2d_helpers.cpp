#include <catch2/catch.hpp>
#include <tensor/nn/conv_2d_helpers.hpp>
#include <tensor/tensor.hpp>

TEST_CASE("_get_flatten_tile(Tensor<float, 3>, ...)")
{
    ts::Tensor<float, 3> tensor = {{{0, 0}, {1, 0}, {2, 0}, {3, 0}},
                                   {{0, 1}, {1, 1}, {2, 1}, {3, 1}},
                                   {{0, 2}, {1, 2}, {2, 2}, {3, 2}},
                                   {{0, 3}, {1, 3}, {2, 3}, {3, 3}}};
    {
        auto result = ts::_get_flatten_tile(tensor, 2, 0, 0);
        auto expected = ts::VectorF{0, 0, 1, 0, 0, 1, 1, 1};
        REQUIRE(result == expected);
    }

    {
        auto result = ts::_get_flatten_tile(tensor, 2, 2, 2);
        auto expected = ts::VectorF{2, 2, 3, 2, 2, 3, 3, 3};
        REQUIRE(result == expected);
    }
}

TEST_CASE("_get_flatten_tile(MatrixF, ...)")
{
    ts::MatrixF matrixF = {{10, 20, 30, 40}, {11, 21, 31, 41}, {12, 22, 32, 42}, {13, 23, 33, 43}};
    {
        auto result = ts::_get_flatten_tile(matrixF, 2, 0, 0);
        auto expected = ts::VectorF{10, 20, 11, 21};
        REQUIRE(result == expected);
    }
    {
        auto result = ts::_get_flatten_tile(matrixF, 2, 0, 1);
        auto expected = ts::VectorF{20, 30, 21, 31};
        REQUIRE(result == expected);
    }
    {
        auto result = ts::_get_flatten_tile(matrixF, 2, 1, 0);
        auto expected = ts::VectorF{11, 21, 12, 22};
        REQUIRE(result == expected);
    }
    {
        auto result = ts::_get_flatten_tile(matrixF, 2, 2, 2);
        auto expected = ts::VectorF{32, 42, 33, 43};
        REQUIRE(result == expected);
    }
    {
        auto result = ts::_get_flatten_tile(matrixF, 3, 0, 0);
        auto expected = ts::VectorF{10, 20, 30, 11, 21, 31, 12, 22, 32};
        REQUIRE(result == expected);
    }
    {
        auto result = ts::_get_flatten_tile(matrixF, 3, 1, 1);
        auto expected = ts::VectorF{21, 31, 41, 22, 32, 42, 23, 33, 43};
        REQUIRE(result == expected);
    }
}
