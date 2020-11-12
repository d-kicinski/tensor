#include <catch2/catch.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace ts;

TEST_CASE("multiply: Matrix[2, 3] X scalar")
{
    Tensor<float, 2> matrix = {{1, 1, 1},
                               {1, 1, 1}};
    float scalar = 1337;

    Tensor<float, 2> expected = {{1337, 1337, 1337},
                                 {1337, 1337, 1337}};
    auto result = multiply(matrix, scalar);

    REQUIRE(result == expected);
}

TEST_CASE("add: Matrix[2, 3] x Matrix[2, 3]")
{
    Matrix t1 = {{1, 1, 1},
                 {1, 1, 1}};
    Matrix t2 = {{1, 1, 1},
                 {1, 1, 1}};
    Matrix expected = {{2, 2, 2},
                       {2, 2, 2}};
    auto result = ts::add(t1, t2);

    REQUIRE(result == expected);
}

TEST_CASE("add: Matrix[2, 3] x Vector[3]")
{
    Matrix matrix = {{1, 1, 1},
                     {0, 0, 0}};
    Vector vector = {3, 3, 3};
    Matrix expected = {{4, 4, 4},
                       {3, 3, 3}};
    auto result = ts::add(matrix, vector);

    REQUIRE(result == expected);
}

TEST_CASE("transpose")
{
    Matrix matrix = {{1, 1, 1},
                     {1, 1, 1}};

    Matrix expected = {{1, 1},
                       {1, 1},
                       {1, 1}};
    auto result = ts::transpose(matrix);

    REQUIRE(result == expected);
}

TEST_CASE("maximum(scalar, Matrix)")
{
    Matrix matrix = {{1, -1, 1},
                     {1, -1, 1}};
    Matrix expected = {{1, 0, 1},
                       {1, 0, 1}};
    auto result = ts::maximum(0.0f, matrix);

    REQUIRE(result == expected);
}

TEST_CASE("mask from tensor")
{
    Matrix matrix = {{1, -1, 1},
                     {1, -1, 1}};
    Tensor<bool, 2> expected = {{true, false, true},
                                {true, false, true}};

    auto mask = ts::mask<float>(matrix, [](float e) { return e >= 0; });

    REQUIRE(mask == expected);
}

TEST_CASE("assign_if")
{
    Matrix matrix = {{1, -1, 1},
                     {1, -1, 1}};
    Matrix expected = {{1, 1337, 1},
                       {1, 1337, 1}};

    auto result = assign_if(matrix, matrix < 0, 1337.0f);

    REQUIRE(result == expected);
}

TEST_CASE("sum")
{
    Matrix matrix = {{1, 1, 1},
                     {1, 1, 1}};
    Vector expected = {2, 2, 2};

    auto result = sum(matrix, 0);

    REQUIRE(result == expected);
}
