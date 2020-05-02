#include "matrix.hpp"
#include <catch2/catch.hpp>

using namespace space;
using namespace std;

TEST_CASE("Construct")
{
    Matrix<int> matrix(3, 3);
    REQUIRE(matrix.shape() == vector{3, 3});
}

TEST_CASE("Construct using list_initialization")
{
    Matrix<int> matrix{{0, 0}, {1, 1}, {2, 2}};
    REQUIRE(matrix.shape() == vector{3, 2});
}

TEST_CASE("Bracket operator")
{
    Matrix<int> matrix{{0, 0}, {1, 1}, {2, 2}};
    REQUIRE(matrix[0] == vector{0, 0});
    REQUIRE(matrix[1] == vector{1, 1});
    REQUIRE(matrix[2] == vector{2, 2});

    REQUIRE(matrix[0][0] == 0);
    REQUIRE(matrix[1][0] == 1);
    REQUIRE(matrix[2][0] == 2);
}

TEST_CASE("Random access via bracket operator")
{
    Matrix<int> matrix{{0, 0}, {1, 1}, {2, 2}};
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

//
// TEST_CASE("operator==") {
//    Tensor1D<int> v1{1, 1, 1, 1};
//    Tensor1D<int> v2{1, 1, 1, 1};
//    REQUIRE(v1 == v2);
//}

// TEST_CASE("operator+") {
//    Tensor1D<int> v1{1, 1, 1, 1};
//    Tensor1D<int> v2{1, 1, 1, 1};
//    Tensor1D<int> v3 = v1 + v2;
//}
