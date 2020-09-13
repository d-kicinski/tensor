#include "flatarray.hpp"
#include <catch2/catch.hpp>
#include <iostream>

template <typename ValueType> auto initialized_array(size_t size, ValueType value) -> ValueType *
{
    auto *array = new ValueType[size];
    for (int i = 0; i < 6; i++) {
        array[i] = value;
    }
    return array;
}

TEST_CASE("simple initialization")
{
    FlatArray<float, 3> flat_array(2, 2, 1);
    FlatArray<float, 1> flat_array2(2, 2, 1);
}

TEST_CASE("not the owner of the data")
{
    std::vector<std::string> vector{"Foo", "Bar", "0", "Spam", "Spam", "1"};
    FlatArray<std::string, 2> array{Dimensions{2, 3}, vector.data()};
}

TEST_CASE("indexing multidimensional array")
{
    float const expected_value = 2.0;
    float *array = initialized_array(6, expected_value);
    FlatArray<float, 2> flat_array{Dimensions{2, 3}, array};

    array[0] = expected_value;
    print(flat_array);

    REQUIRE(expected_value == array[0]);
    REQUIRE(expected_value == flat_array(0, 0));

    array[2 * 3 - 1] = expected_value;
    print(flat_array);

    REQUIRE(expected_value == array[2 * 3 - 1]);
    REQUIRE(expected_value == flat_array(1, 2));
}

TEST_CASE("shape: simple case")
{
    FlatArray<int, 2> matrix{Dimensions{2, 3}, initialized_array(6, 1)};
    std::vector<int> expected_shape{2, 3};
    REQUIRE(expected_shape == matrix.shape());
}

TEST_CASE("shape: scalar")
{
    FlatArray<int, 1> matrix{Dimensions{1}, initialized_array(1, 1)};
    std::vector<int> expected_shape{1};
    REQUIRE(expected_shape == matrix.shape());
}

TEST_CASE("operator[]")
{
    FlatArray<int, 2> matrix{Dimensions{2, 3}, initialized_array(6, 1)};
    print(matrix.dimensions(), 2);

    FlatArray<int, 1, false> array = matrix[0];
    std::vector<int> expected_shape{3};
    REQUIRE(expected_shape == array.shape());
    REQUIRE(1 == array[0]);
}

TEST_CASE("operator==")
{
    FlatArray<int, 1> array{Dimensions{6}, initialized_array(6, 1)};
    FlatArray<int, 1> array_same{Dimensions{6}, initialized_array(6, 1)};
    FlatArray<int, 1> array_different_value{Dimensions{6}, initialized_array(6, 2)};
    FlatArray<int, 1> array_different_shape{Dimensions{8}, initialized_array(8, 1)};

    REQUIRE((array_same == array));
    REQUIRE((array_different_value != array));
    REQUIRE((array_different_shape != array));
}
