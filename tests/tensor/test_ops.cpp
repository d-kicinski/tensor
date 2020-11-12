#include <catch2/catch.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace ts;

TEST_CASE("multiply: Matrix[2, 3] X Vector[3]")
{
    Matrix matrix = {{1, 1, 1},
                     {2, 2, 2}};
    Vector vector = {1, 1, 1};
    Vector result = multiply(matrix, vector);
    Vector expected_result = {3, 6};

    REQUIRE(expected_result == result);
}


TEST_CASE("multiply: Matrix[3, 3] X Vector[3]")
{
    Matrix matrix = {
        {3, 1, 3},
        {1, 5, 9},
        {2, 6, 5}
    };
    Vector vector= { -1, -1, 1 };
    Vector result = multiply(matrix, vector);
    Vector expected_result = {-1, 3, -3};

    REQUIRE(expected_result == result);
}

TEST_CASE("multiply: Matrix[3, 3] X Matrix[3, 3]")
{
    Matrix matrix = {
        {3, 1, 3},
        {1, 5, 9},
        {2, 6, 5}
    };
    Matrix result = multiply(matrix, matrix);
    Matrix expected_result = {
        {16, 26, 33},
        {26, 80, 93},
        {22, 62, 85}
    };

    REQUIRE(expected_result == result);
}

TEST_CASE("multiply: Matrix[2, 3] X Matrix[3, 2]")
{
    Matrix matrixA = {
        {3, 1, 3},
        {1, 5, 9},
    };
    Matrix matrixB = {
        {3, 1},
        {1, 5},
        {2, 6}
    };
    Matrix result = multiply(matrixA, matrixB);
    Matrix expected_result = {
        {16, 26},
        {26, 80},
    };

    REQUIRE(expected_result == result);
}

TEST_CASE("multiply: Matrix[3, 2].T X Matrix[3, 2]")
{
    Matrix matrixA = {
        {3, 1},
        {1, 5},
        {3, 9}
    };
    Matrix matrixB = {
        {3, 1},
        {1, 5},
        {2, 6}
    };
    Matrix result = multiply(matrixA, matrixB, true);
    Matrix expected_result = {
        {16, 26},
        {26, 80},
    };

    REQUIRE(expected_result == result);
}

TEST_CASE("multiply: Matrix[2, 3] X Matrix[2, 3].T")
{
    Matrix matrixA = {
        {3, 1, 3},
        {1, 5, 9},
    };

    Matrix matrixB = {
        {3, 1, 2},
        {1, 5, 6}
    };

    Matrix result = multiply(matrixA, matrixB, false, true);
    Matrix expected_result = {
        {16, 26},
        {26, 80},
    };

    REQUIRE(expected_result == result);
}

TEST_CASE("add")
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

TEST_CASE("multiply: Tensor[2, 2, 2] x Matrix[2, 3]")
{
    Tensor<float, 3> tensorA = {
        {{1, 1},
         {2, 2}},

        {{3, 3},
         {4, 4}}
    };
    Matrix matrixB = {
        {3, 1, 3},
        {1, 5, 9},
    };
    Tensor<float, 3> expected = {
        {{4, 6, 12},
         {8, 12, 24}},

        {{12, 18, 36},
         {16, 24, 48}}
    };
    Tensor<float, 3> result = ts::multiply(tensorA, matrixB);

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
