#include "tensor1d.hpp"
#include <catch2/catch.hpp>

using namespace space;

TEST_CASE("Construct") {
    Tensor1D<int> vector(8);
    REQUIRE(vector.length == 8);
}

TEST_CASE("Assign via bracket operator") {
    Tensor1D<int> vector(8);
    int const value = 1;
    vector[0] = value;
    REQUIRE(vector[0] == value);
}

TEST_CASE("Construct using list_initialization") {
    Tensor1D<int> vector{1, 1, 1, 1};
    REQUIRE(vector.length == 4);
}

TEST_CASE("operator==") {
    Tensor1D<int> v1{1, 1, 1, 1};
    Tensor1D<int> v2{1, 1, 1, 1};
    REQUIRE(v1 == v2);
}

// TEST_CASE("operator+") {
//    Tensor1D<int> v1{1, 1, 1, 1};
//    Tensor1D<int> v2{1, 1, 1, 1};
//    Tensor1D<int> v3 = v1 + v2;
//}