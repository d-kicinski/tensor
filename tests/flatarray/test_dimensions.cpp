#include "flatarray/dimensions.hpp"
#include <catch2/catch.hpp>

TEST_CASE("create simple Dimensions object") {
    Dimensions dimensions (2, 3);
    assert(dimensions.data_size == 6);
}