#include <catch2/catch.hpp>
#include "dimensions.hpp"

TEST_CASE("create simple Dimensions object") {
    Dimensions dimensions (2, 3);
    assert(dimensions.data_size == 6);
}