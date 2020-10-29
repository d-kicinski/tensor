#include <catch2/catch.hpp>

#include "tensor/ops.hpp"

using namespace ts;

TEST_CASE("dot: Matrix[2, 3] X Vector[3]")
{
    Matrix matrix = {{1, 1, 1},
                     {2, 2, 2}};
    Vector vector = {1, 1, 1};
    Vector result = dot(matrix, vector);
    Vector expected_result = {3, 6};

    REQUIRE(expected_result == result);
}


TEST_CASE("dot: Matrix[3, 3] X Vector[3]")
{
    Matrix matrix = {
        {3, 1, 3},
        {1, 5, 9},
        {2, 6, 5}
    };
    Vector vector= { -1, -1, 1 };
    Vector result = dot(matrix, vector);
    Vector expected_result = {-1, 3, -3};

    REQUIRE(expected_result == result);
}

TEST_CASE("dot: Matrix[3, 3] X Matrix[3, 3]")
{
    Matrix matrix = {
        {3, 1, 3},
        {1, 5, 9},
        {2, 6, 5}
    };
    Matrix result = dot(matrix, matrix);
    Matrix expected_result = {
        {16, 26, 33},
        {26, 80, 93},
        {22, 62, 85}
    };

    REQUIRE(expected_result == result);
}

TEST_CASE("dot: Matrix[2, 3] X Matrix[3, 2]")
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
    Matrix result = dot(matrixA, matrixB);
    Matrix expected_result = {
        {16, 26},
        {26, 80},
    };

    REQUIRE(expected_result == result);
}
