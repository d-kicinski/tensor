#include <catch2/catch.hpp>
#include "tensor1d.hpp"

using namespace space;

TEST_CASE("Construct") {
    Tensor1D<int> vector(8);
    REQUIRE( vector.length == 8 );
}

TEST_CASE("Assign via bracket operator") {
   Tensor1D<int> vector (8);
   int const value = 1;
   vector[0] = value;
   REQUIRE( vector[0] == value );
}