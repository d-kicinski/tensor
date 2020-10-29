#include "tensor/dimensions.hpp"
#include <catch2/catch.hpp>

using namespace ts;

TEST_CASE("create simple Dimensions object") {
    Dimensions dimensions (2, 3);
    assert(dimensions.data_size == 6);
}