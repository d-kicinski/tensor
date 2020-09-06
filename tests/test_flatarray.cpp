#include <catch2/catch.hpp>
#include "flatarray.hpp"

TEST_CASE("simple initialization") {
   FlatArray<float, 3> flat_array(2, 2, 1);

   // hmmm?
   FlatArray<float, 1> flat_array2(2, 2, 1);
}