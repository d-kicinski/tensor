#include "flatarray.hpp"
#include <catch2/catch.hpp>
#include <iostream>

TEST_CASE("simple initialization") {
   FlatArray<float, 3> flat_array(2, 2, 1);

   // hmmm?
   FlatArray<float, 1> flat_array2(2, 2, 1);
}

TEST_CASE("not the owner of the data") {
    std::vector<std::string> vector {"Foo", "Bar", "0", "Spam", "Spam", "1"};
    FlatArray<std::string, 2> array {Dimensions{2, 3}, vector.data()};
}

TEST_CASE("indexing multidimensional array") {
    auto * array = new float[6];
    for (int i = 0; i<6; i++) {
        array[i] = 1.0;
    }
    FlatArray<float, 2> flat_array {Dimensions{2, 3}, array};

    float const expected_value = 2.0;

    array[0] = expected_value;
    print_flat_array(flat_array);

    REQUIRE(expected_value == array[0]);
    REQUIRE(expected_value == flat_array(0, 0));

    array[2*3 - 1] = expected_value;
    print_flat_array(flat_array);

    REQUIRE(expected_value == array[2*3 - 1]);
    REQUIRE(expected_value == flat_array(1, 2));
}

