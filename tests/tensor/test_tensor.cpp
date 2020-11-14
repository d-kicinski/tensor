#include <iostream>
#include <catch2/catch.hpp>
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

auto get_2_by_3_float_tensor() -> Tensor<float, 2>
{
    return {{1, 2, 3},
            {4, 5, 6}};
}

TEST_CASE("simple initialization")
{
    Tensor<float, 3> flat_array(2, 2, 1);
    REQUIRE(flat_array.shape() == std::array{2, 2, 1});
}

TEST_CASE("not the owner of the data")
{
//    std::vector<std::string> vector{"Foo", "Bar", "0", "Spam", "Spam", "1"};
//    Tensor<std::string, 2> array{Dimensions{2, 3}, vector.data()};
//    REQUIRE(array.shape() == std::vector{2, 3});
}

TEST_CASE("simple initializer_list")
{
    Tensor<int, 1> array = {0, 1, 2, 3};
    REQUIRE(array.shape() == std::array{4});
    REQUIRE(array.data_size() == 4);
    REQUIRE(array[3] == 3);
}

TEST_CASE("nested initializer_list")
{
    Tensor<int, 2> array = {{0, 1},
                            {2, 3}};
    REQUIRE(array.shape() == std::array{2, 2});
    REQUIRE(array.data_size() == 4);
    REQUIRE(array.data()->at(0) == 0);
    REQUIRE(array.data()->at(1) == 1);
    REQUIRE(array.data()->at(2) == 2);
    REQUIRE(array.data()->at(3) == 3);
}

TEST_CASE("indexing multidimensional array")
{
    int const expected_value = 2;
    Tensor<int, 2> tensor(2, 3);

    tensor.data()->at(0) = expected_value;
    REQUIRE(expected_value == tensor(0, 0));

    tensor.data()->at(2 * 3 - 1) = expected_value;
    REQUIRE(expected_value == tensor(1, 2));
}

TEST_CASE("shape: simple case")
{
    Tensor<int, 2> matrix(2, 3);
    REQUIRE(matrix.shape() == std::array{2, 3});
}

TEST_CASE("shape: scalar")
{
    Tensor<int, 1> matrix = {1};
    REQUIRE(matrix.shape() == std::array{1});
}

TEST_CASE("bracket operator")
{
    Tensor<float, 2> matrix = get_2_by_3_float_tensor();
    Tensor<float, 1> array = matrix[0];

    REQUIRE(matrix.shape() == std::array{2, 3});
    REQUIRE(array.shape() == std::array{3});
    REQUIRE(1 == array[0]);
}

TEST_CASE("operator==")
{
    Tensor<int, 1> array = {0, 0};
    Tensor<int, 1> array_same = {0, 0};
    Tensor<int, 1> array_different_value = {1, 1};
    Tensor<int, 1> array_different_shape = {0, 0, 0};

    REQUIRE((array_same == array));
    REQUIRE((array_different_value != array));
    REQUIRE((array_different_shape != array));
}

TEST_CASE("basic iterator")
{
   Tensor<int, 1> tensor = {0, 1, 2, 3, 4, 5};
   int expected_element = 0;
   for (auto element: tensor) {
       REQUIRE(expected_element == element);
       ++expected_element;
   }
}

TEST_CASE("copy constructor")
{
    Tensor<int, 2> t1 = {{1, 2, 3}, { 4, 5, 6}};
    Tensor<int, 2> t2(t1);

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
    REQUIRE(matrix.shape() == std::array{2, 2});
}

TEST_CASE("sub-array indexing")
{
   Tensor<float, 2> tensor = {{0, 0},
                              {1, 1},
                              {2, 2}};

   Tensor<float, 1> v1 = tensor(0);
   Tensor<float, 1> v2 = tensor(1);
   Tensor<float, 1> v3 = tensor(2);

   Tensor<float, 1> expected_v1 = {0, 0};
   Tensor<float, 1> expected_v2 = {1, 1};
   Tensor<float, 1> expected_v3 = {2, 2};

   REQUIRE(v1 == expected_v1);
   REQUIRE(v2 == expected_v2);
   REQUIRE(v3 == expected_v3);
}

TEST_CASE("access element of a sub-array")
{
    Tensor<float, 3> tensor = {
        {{0, 0},
         {1, 1}},
        {{2, 2},
         {3, 3}}
    };
    Tensor<float, 2> matrix = tensor(1);
    Tensor<float, 2> expected = {{2, 2},
                                 {3, 3}};

    REQUIRE(matrix == expected);
    REQUIRE(matrix(0, 0) == expected(0, 0));
    REQUIRE(matrix(1, 1) == expected(1, 1));
}

TEST_CASE("unary operator-")
{
   Matrix matrix = {{1, 1, 1},
                    {1, 1, 1}};
   Matrix expected = {{-1, -1, -1},
                      {-1, -1, -1}};
   Matrix result = -matrix;

   REQUIRE(result == expected);
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
