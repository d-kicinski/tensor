#include <catch2/catch.hpp>
#include "flatarray.hpp"

TEST_CASE("simple initialization") {
   FlatArray<float, 3> flat_array(2, 2, 1);

   // hmmm?
   FlatArray<float, 1> flat_array2(2, 2, 1);
}

TEST_CASE("not the owner of the data") {
    std::vector<std::string> vector {"Foo", "Bar", "0", "Spam", "Spam", "1"};
    FlatArray<std::string, 2> array {Dimensions{2, 3}, vector.data()};
}