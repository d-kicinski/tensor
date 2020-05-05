#include "matrix.hpp"
#include <catch2/catch.hpp>

using namespace space;
using namespace std;

TEST_CASE("Construct")
{
    Matrix matrix(3, 3);
    REQUIRE(matrix.shape() == vector<unsigned int>{3, 3});
}

TEST_CASE("Construct with initializer list")
{
    Matrix matrix{3, 3};
    REQUIRE(matrix.shape() == vector<unsigned int>{3, 3});
}

TEST_CASE("Construct using list_initialization")
{
    Matrix matrix{{0, 0}, {1, 1}, {2, 2}};
    REQUIRE(matrix.shape() == vector<unsigned int>{3, 2});
}

TEST_CASE("Bracket operator")
{
    Matrix matrix{{0, 0}, {1, 1}, {2, 2}};
    REQUIRE(matrix[0] == vector<float>{0, 0});
    REQUIRE(matrix[1] == vector<float>{1, 1});
    REQUIRE(matrix[2] == vector<float>{2, 2});

    REQUIRE(matrix[0][0] == 0);
    REQUIRE(matrix[1][0] == 1);
    REQUIRE(matrix[2][0] == 2);
}

TEST_CASE("Random access via bracket operator")
{
    Matrix matrix{{0, 0}, {1, 1}, {2, 2}};
    REQUIRE(matrix[{0, 0}] == 0);
    REQUIRE(matrix[{0, 1}] == 0);
    REQUIRE(matrix[{1, 0}] == 1);
    REQUIRE(matrix[{1, 1}] == 1);
    REQUIRE(matrix[{2, 0}] == 2);
    REQUIRE(matrix[{2, 1}] == 2);

    matrix[{0, 1}] = 100;
    matrix[{1, 1}] = 100;
    matrix[{2, 1}] = 100;

    REQUIRE(matrix[{0, 1}] == 100);
    REQUIRE(matrix[{1, 1}] == 100);
    REQUIRE(matrix[{2, 1}] == 100);
}

TEST_CASE("Constructs by assigment")
{
    Matrix m1 = {{1, 1}};
    REQUIRE(m1[{0, 0}] == 1);
    REQUIRE(m1[{0, 1}] == 1);
}

TEST_CASE("operator==")
{
    Matrix m1 = {{1, 1}, {1, 1}};
    Matrix m2 = {{1, 1}, {1, 1}};
    REQUIRE(m1 == m2);
}

TEST_CASE("Static initialize with zeros")
{
    auto m1 = Matrix::zeros(2, 2);
    auto m2 = Matrix::zeros(2, 2);
    REQUIRE(m1 == m2);
}

TEST_CASE("Static initialize with ones")
{
    auto m1 = Matrix::ones(2, 2);
    Matrix m2 = {{1, 1}, {1, 1}};
    REQUIRE(m1 == m2);
}

TEST_CASE("operator+")
{
    Matrix m1 = {{1, 1}};
    Matrix m2 = {{1, 1}};
    Matrix result = m1 + m2;

    Matrix expected = {{2, 2}};
    REQUIRE(expected == result);
}

TEST_CASE("Iterate")
{
//    Matrix matrix = {{1, 1}, {2, 2}};
//    for (MatrixView matrixView : matrix) {
//        // pass
//    }
}

// TEST_CASE("operator@")
//{
//    Matrix<int> m1 = {{1, 2}, {3, 4}, {5, 6}};
//    Matrix<int> m2 = {{1, 2, 3}, {4, 5, 6}};
//    Matrix<int> result = multiply(m1, m2);
//
//    Matrix<int> expected = {{9, 12, 15}, {19, 26, 33}, {29, 40, 51}};
//    REQUIRE(expected == result);
//}
