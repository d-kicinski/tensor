#include <iostream>
#include <catch2/catch.hpp>
#include <tensor/dimensions.hpp>
#include <tensor/tensor.hpp>

using namespace ts;

template <typename ValueType> auto initialized_array(size_t size, ValueType value) -> ValueType *
{
    auto *array = new ValueType[size];
    for (int i = 0; i < size; i++) {
        array[i] = value;
    }
    return array;
}

TEST_CASE("simple initialization")
{
    Tensor<float, 3> flat_array(2, 2, 1);
    REQUIRE(flat_array.shape() == std::vector{2, 2, 1});
}

TEST_CASE("not the owner of the data")
{
    std::vector<std::string> vector{"Foo", "Bar", "0", "Spam", "Spam", "1"};
    Tensor<std::string, 2> array{Dimensions{2, 3}, vector.data()};
    REQUIRE(array.shape() == std::vector{2, 3});
}

TEST_CASE("simple initializer_list")
{
    Tensor<int, 1> array = {0, 1, 2, 3};
    REQUIRE(array.shape() == std::vector{4});
    REQUIRE(array.data_size() == 4);
    REQUIRE(array.data()[3] == 3);
}

TEST_CASE("nested initializer_list")
{
    Tensor<int, 2> array = {{0, 1},
                               {2, 3}};
    REQUIRE(array.shape() == std::vector{2, 2});
    REQUIRE(array.data_size() == 4);
    REQUIRE(array.data()[0] == 0);
    REQUIRE(array.data()[1] == 1);
    REQUIRE(array.data()[2] == 2);
    REQUIRE(array.data()[3] == 3);
}

TEST_CASE("indexing multidimensional array")
{
    int const expected_value = 2;
    int *array = initialized_array(6, 0);
    Tensor<int, 2> flat_array(Dimensions{2, 3}, array);

    array[0] = expected_value;
    REQUIRE(expected_value == array[0]);
    REQUIRE(expected_value == flat_array(0, 0));

    array[2 * 3 - 1] = expected_value;
    REQUIRE(expected_value == array[2 * 3 - 1]);
    REQUIRE(expected_value == flat_array(1, 2));
}

TEST_CASE("shape: simple case")
{
    Tensor<int, 2> matrix{Dimensions{2, 3}, initialized_array(6, 1)};
    std::vector<int> expected_shape{2, 3};
    REQUIRE(expected_shape == matrix.shape());
}

TEST_CASE("shape: scalar")
{
    Tensor<int, 1> matrix{Dimensions{1}, initialized_array(1, 1)};
    std::vector<int> expected_shape{1};
    REQUIRE(expected_shape == matrix.shape());
}

TEST_CASE("bracket operator")
{
    Tensor<int, 2> matrix{Dimensions{3, 2}, initialized_array(6, 1)};
    Tensor<int, 1> array = matrix[0];
    std::vector<int> expected_shape{2};
    REQUIRE(expected_shape == array.shape());
    REQUIRE(1 == array[0]);
}

TEST_CASE("operator==")
{
    Tensor<int, 1> array{Dimensions{6}, initialized_array(6, 1)};
    Tensor<int, 1> array_same{Dimensions{6}, initialized_array(6, 1)};
    Tensor<int, 1> array_different_value{Dimensions{6}, initialized_array(6, 2)};
    Tensor<int, 1> array_different_shape{Dimensions{8}, initialized_array(8, 1)};

    REQUIRE((array_same == array));
    REQUIRE((array_different_value != array));
    REQUIRE((array_different_shape != array));
}

TEST_CASE("basic iterator")
{
   int array[6] = {0, 1, 2, 3, 4, 5};
   Tensor<int, 1> flat_array(Dimensions{1}, array);
   int expected_element = 0;
   for (auto element: flat_array) {
       REQUIRE(expected_element == element);
       ++expected_element;
   }
}

TEST_CASE("copy constructor")
{
    Tensor<int, 2> t1 = {{1, 2, 3}, { 4, 5, 6}};
    Tensor<int, 2> t2(t1);

    REQUIRE(t2.owner() == true);
    REQUIRE(t1 == t2);

    t2(0, 0) = 1337;
    REQUIRE(t1(0, 0) == 1);
    REQUIRE(t2(0, 0) == 1337);
}

TEST_CASE("construct from shape")
{
    Tensor<int, 2> t1 = {{1, 2, 3},
                         {4, 5, 6}};
    Tensor<int, 2> t2(t1.shape());

    REQUIRE(t2.owner() == true);
    REQUIRE(t2.shape() == t1.shape());
    REQUIRE(t2.data_size() == t2.data_size());
}

TEST_CASE("operator<(Tensor, scalar)")
{

    Matrix matrix = {{1, -1, 1},
                     {1, -1, 1}};
    auto result = matrix > 0;
    Tensor<bool, 2> expected = {{true, false, true},
                                {true, false, true}};
    REQUIRE(result == expected);
}

TEST_CASE("Assign to existing matrix")
{
    Matrix matrix;
    matrix = Matrix(2, 2);
}

TEST_CASE("random tensor")
{
    auto matrix = Tensor<float, 2>::randn({2, 2});
    std::vector<int> expected_shape = {2, 2};
    REQUIRE(matrix.shape() == expected_shape);
}

TEST_CASE("sub-array iterator 2D")
{
//    int array[6] = {0, 0, 1, 1, 2, 2};
//    Tensor<int, 2> flat_array(Dimensions{3, 2}, array);
//    int value = 0;
//    for(auto element: flat_array) {
//        int* expected_subarray = initialized_array(2, value);
//        REQUIRE(Tensor<int, 1>(Dimensions{1}, expected_subarray) == element);
//        ++value;
//    }
}
