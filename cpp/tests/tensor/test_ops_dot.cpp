#include <catch2/catch.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>

using namespace ts;

TEST_CASE("outer_product")
{
    VectorF x = {0, 1, 2, 3};
    VectorF y = {2, -2};
    MatrixF result = ts::outer_product(x, y);
    MatrixF expected = {{0, 0},
                        {2, -2},
                        {4, -4},
                        {6, -6}};
    REQUIRE(result == expected);

}

TEST_CASE("dot: VectorF[3] X VectorF[3]")
{
    VectorF a = {1, 2, 3};
    VectorF b = {10, 10, 10};
    float result = dot(a, b);
    float expected_result = 60;
    REQUIRE(expected_result == result);
}


TEST_CASE("dot: MatrixF[2, 3] X VectorF[3]")
{
    MatrixF matrix = {{1, 1, 1},
                     {2, 2, 2}};
    VectorF vector = {1, 1, 1};
    VectorF result = dot(matrix, vector);
    VectorF expected_result = {3, 6};

    REQUIRE(expected_result == result);
}


TEST_CASE("dot: MatrixF[3, 3] X VectorF[3]")
{
    MatrixF matrix = {
        {3, 1, 3},
        {1, 5, 9},
        {2, 6, 5}
    };
    VectorF vector= { -1, -1, 1 };
    VectorF result = dot(matrix, vector);
    VectorF expected_result = {-1, 3, -3};

    REQUIRE(expected_result == result);
}

TEST_CASE("dot: MatrixF[3, 3] X MatrixF[3, 3]")
{
    MatrixF matrix = {
        {3, 1, 3},
        {1, 5, 9},
        {2, 6, 5}
    };
    MatrixF result = dot(matrix, matrix);
    MatrixF expected_result = {
        {16, 26, 33},
        {26, 80, 93},
        {22, 62, 85}
    };

    REQUIRE(expected_result == result);
}

TEST_CASE("dot: MatrixF[2, 3] X MatrixF[3, 2]")
{
    MatrixF matrixA = {
        {3, 1, 3},
        {1, 5, 9},
    };
    MatrixF matrixB = {
        {3, 1},
        {1, 5},
        {2, 6}
    };
    MatrixF result = dot(matrixA, matrixB);
    MatrixF expected_result = {
        {16, 26},
        {26, 80},
    };

    REQUIRE(expected_result == result);
}

TEST_CASE("dot: MatrixF[3, 2].T X MatrixF[3, 2]")
{
    MatrixF matrixA = {
        {3, 1},
        {1, 5},
        {3, 9}
    };
    MatrixF matrixB = {
        {3, 1},
        {1, 5},
        {2, 6}
    };
    MatrixF result = dot(matrixA, matrixB, true);
    MatrixF expected_result = {
        {16, 26},
        {26, 80},
    };

    REQUIRE(expected_result == result);
}

TEST_CASE("dot: MatrixF[2, 3] X MatrixF[2, 3].T")
{
    MatrixF matrixA = {
        {3, 1, 3},
        {1, 5, 9},
    };

    MatrixF matrixB = {
        {3, 1, 2},
        {1, 5, 6}
    };

    MatrixF result = dot(matrixA, matrixB, false, true);
    MatrixF expected_result = {
        {16, 26},
        {26, 80},
    };

    REQUIRE(expected_result == result);
}

TEST_CASE("dot: Tensor[2, 2, 2] x MatrixF[2, 3]")
{
    Tensor<float, 3> tensorA = {
        {{1, 1},
         {2, 2}},

        {{3, 3},
         {4, 4}}
    };
    MatrixF matrixB = {
        {3, 1, 3},
        {1, 5, 9},
    };
    Tensor<float, 3> expected = {
        {{4, 6, 12},
         {8, 12, 24}},

        {{12, 18, 36},
         {16, 24, 48}}
    };
    Tensor<float, 3> result = ts::dot(tensorA, matrixB);

    REQUIRE(result == expected);
}
