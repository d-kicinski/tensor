#include <iostream>
#include <catch2/catch.hpp>
#include <tensor/tensor.hpp>

using namespace ts;

TEST_CASE("simple initialization")
{
    Tensor<float, 3> tensor(2, 2, 1);
    REQUIRE(tensor.shape() == std::array<size_type, 3>{2, 2, 1});
    REQUIRE(tensor.data_size() == 4);
}

TEST_CASE("simple initializer_list")
{
    Tensor<int, 1> array = {0, 1, 2, 3};
    REQUIRE(array.shape() == std::array<size_type, 1>{4});
    REQUIRE(array.data_size() == 4);
    REQUIRE(*array.data() == std::vector{0, 1, 2, 3});
}

TEST_CASE("nested initializer_list")
{
    Tensor<int, 2> array = {{0, 1},
                            {2, 3}};
    REQUIRE(array.shape() == std::array<size_type, 2>{2, 2});
    REQUIRE(array.data_size() == 4);
    REQUIRE(*array.data() == std::vector{0, 1, 2, 3});
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
    REQUIRE(matrix.shape() == std::array<size_type, 2>{2, 3});
}

TEST_CASE("shape: scalar")
{
    Tensor<int, 1> matrix = {1};
    REQUIRE(matrix.shape() == std::array<size_type, 1>{1});
}

TEST_CASE("shape(index)")
{
    Tensor<int, 2> matrix(2, 3);
    REQUIRE(matrix.shape(0) == 2);
    REQUIRE(matrix.shape(1) == 3);
}

TEST_CASE("bracket operator")
{
    Tensor<float, 2> matrix ={{1, 2, 3},
                              {4, 5, 6}};
    REQUIRE(matrix.shape() == std::array<size_type, 2>{2, 3});
    REQUIRE(*matrix.data() == std::vector<float>{1, 2, 3, 4, 5, 6});

    {
        Tensor<float, 1> array = matrix[0];
        REQUIRE(array.shape() == std::array<size_type, 1>{3});
        std::vector<float> expected = {1, 2, 3};
        REQUIRE(std::equal(array.begin(), array.end(), expected.begin()));
        REQUIRE(array[0] == 1);
        REQUIRE(array[1] == 2);
        REQUIRE(array[2] == 3);
    }
    {
        Tensor<float, 1> array = matrix[1];
        REQUIRE(array.shape() == std::array<size_type, 1>{3});
        std::vector<float> expected = {4, 5, 6};
        REQUIRE(std::equal(array.begin(), array.end(), expected.begin()));
        REQUIRE(array[0] == 4);
        REQUIRE(array[1] == 5);
        REQUIRE(array[2] == 6);
    }
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

TEST_CASE("iterator of sub-array")
{
    Tensor<float, 2> matrix ={{1, 2, 3},
                              {4, 5, 6}};
    Tensor<float, 1> array = matrix(1);
    int first_expected = 4;
    for (auto element: array) {
        REQUIRE(first_expected == element);
        ++first_expected;
    }
}


TEST_CASE("copy constructor: should create an alias")
{
    Tensor<float, 2> matrix = {{1, 2, 3},
                               {4, 5, 6}};
    Tensor<float, 2> matrix_copy(matrix);
    REQUIRE(matrix == matrix_copy);
    REQUIRE(matrix.shape() == matrix_copy.shape());
    REQUIRE(matrix.data_size() == matrix_copy.data_size());
    REQUIRE(*matrix.data() == *matrix_copy.data());

    matrix_copy(0, 0) = 1337.0f;
    REQUIRE(matrix_copy(0, 0) == 1337.0f);
    REQUIRE(matrix(0, 0) == 1337.0f);
}


TEST_CASE("assigment operator: should create an alias")
{
    Tensor<float, 2> matrix = {{1, 2, 3},
                               {4, 5, 6}};
    Tensor<float, 2> matrix_alias;
    matrix_alias = matrix;

    REQUIRE(matrix.shape() == matrix_alias.shape());
    REQUIRE(matrix.data_size() == matrix_alias.data_size());
    REQUIRE(*matrix.data() == *matrix_alias.data());

    matrix(0, 0) = 1337.0f;
    REQUIRE(matrix(0, 0) == 1337.0f);
    REQUIRE(matrix_alias(0, 0) == 1337.0f);
}

TEST_CASE("clone()")
{
    Tensor<float, 2> matrix = {{1, 2, 3},
                               {4, 5, 6}};
    Tensor<float, 2> matrix_copy = matrix.clone();

    REQUIRE(matrix.shape() == matrix_copy.shape());
    REQUIRE(matrix.data_size() == matrix_copy.data_size());
    REQUIRE(*matrix.data() == *matrix_copy.data());

    matrix(0, 0) = 1337.0f;
    REQUIRE(matrix(0, 0) == 1337.0f);
    REQUIRE(matrix_copy(0, 0) == 1.0f);
}

TEST_CASE("construct from shape")
{
    Tensor<int, 2> t1 = {{1, 2, 3},
                         {4, 5, 6}};
    Tensor<int, 2> t2(t1.shape());

    REQUIRE(t2.shape() == t1.shape());
    REQUIRE(t2.data_size() == t2.data_size());
    REQUIRE(!std::equal(t1.begin(), t1.end(), t2.begin()));
}

TEST_CASE("operator<(Tensor, scalar)")
{
    MatrixF matrix = {{1, -1, 1},
                     {1, -1, 1}};
    auto result = matrix > 0;
    Tensor<char, 2> expected = {{true, false, true},
                                {true, false, true}};
    REQUIRE(result == expected);
}

TEST_CASE("rand(shape)")
{
    auto matrix = Tensor<float, 2>::randn({2, 2});
    REQUIRE(matrix.shape() == std::array<size_type, 2>{2, 2});
    REQUIRE(matrix.data_size() == 4);
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
   MatrixF matrix = {{1, 1, 1},
                    {1, 1, 1}};
   MatrixF expected = {{-1, -1, -1},
                      {-1, -1, -1}};
   MatrixF result = -matrix;

   REQUIRE(result == expected);
}

TEST_CASE("cast")
{
   Tensor<float, 2> matrix_f = {{1.0f, 2.0f},
                                {3.0f, 4.0f}};
   Tensor<int, 2> matrix_i = {{1, 2},
                              {3, 4}};

   REQUIRE(matrix_i == matrix_f.cast<int>());
}

TEST_CASE("flatten: MatrixF -> VectorF")
{
    MatrixF matrix = {{1, 2},
                      {3, 4}};
    VectorF expected = {1, 2, 3, 4};
    auto result = matrix.flatten();

    REQUIRE(result == expected);
}

TEST_CASE("flatten: TensorF -> VectorF")
{
    Tensor<float, 3> tensor = {
        {{1, 2},
         {3, 4}},
        {{5, 6},
         {7, 8}}
    };
    REQUIRE(tensor.data().use_count() == 2);
    {
        VectorF expected = {1, 2, 3, 4, 5, 6, 7, 8};
        auto result = tensor.flatten();
        REQUIRE(tensor.data().use_count() == 3);
        REQUIRE(result == expected);
    }
    REQUIRE(tensor.data().use_count() == 2);
    {
        auto matrix = tensor(1);
        REQUIRE(tensor.data().use_count() == 3);
        VectorF expected = {5, 6, 7, 8};
        auto result = matrix.flatten();
        REQUIRE(tensor.data().use_count() == 4);
        REQUIRE(result == expected);
    }
}

TEST_CASE("reshape")
{
    Tensor<int, 3> tensor3 = {
        {{0, 1}, {2, 3}},
        {{0, 1}, {2, 3}},
        {{0, 1}, {2, 3}},
    };
    REQUIRE(tensor3.shape() == std::array<size_type, 3>{3, 2, 2});

    Tensor<int, 2> tensor2 = tensor3.reshape<2>({3, 4});
    Tensor<int, 2> expected_tensor2 = {
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {0, 1, 2, 3},
    };
    REQUIRE(tensor2.shape() == std::array<size_type, 2>{3, 4});
    REQUIRE(tensor2 == expected_tensor2);

    Tensor<int, 1> tensor1 = tensor3.reshape<1>({12});
    Tensor<int, 1> expected_tensor1 = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    REQUIRE(tensor1.shape() == std::array<size_type, 1>{12});
    REQUIRE(tensor1 == expected_tensor1);
}

TEST_CASE("at")
{
    ts::Tensor<float, 3> tensor = {{{0, 1}, {2, 3}, {4, 5}}, {{6, 7}, {8, 9}, {10, 11}}};
    REQUIRE(tensor.at({0, 0, 0}) == 0);
    REQUIRE(tensor.at({1, 1, 1}) == 9);
    REQUIRE(tensor.at({1, 2, 1}) == 11);
}

TEST_CASE("get_subarray") {
    ts::Tensor<float, 3> tensor = {{{0, 1}, {2, 3}, {4, 5}}, {{6, 7}, {8, 9}, {10, 11}}};
    {
        auto [b, e] = tensor.get_subarray({0, 0});

        REQUIRE(std::equal(b, e, tensor(0, 0).begin()));
    }
    {
        auto [b, e] = tensor.get_subarray({1, 1});
        REQUIRE(std::equal(b, e, tensor(1, 1).begin()));
    }
    {
        auto [b, e] = tensor.get_subarray({1, 2});
        REQUIRE(std::equal(b, e, tensor(1, 2).begin()));
    }
}
