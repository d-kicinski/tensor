#include <catch2/catch.hpp>
#include <tensor/dimensions.hpp>

using namespace ts;

TEST_CASE("create simple Dimensions object") {
    Dimensions dimensions (2, 3);
    assert(dimensions.data_size == 6);
}