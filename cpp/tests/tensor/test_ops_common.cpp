#include <functional>
#include <cmath>

#include <catch2/catch.hpp>
#include <tensor/ops.hpp>
#include <tensor/tensor.hpp>


using namespace ts;
using namespace std::placeholders;

TEST_CASE("multiply: MatrixF[2, 3] X scalar")
{
    Tensor<float, 2> matrix = {{1, 1, 1},
                               {1, 1, 1}};
    float scalar = 1337;

    Tensor<float, 2> expected = {{1337, 1337, 1337},
                                 {1337, 1337, 1337}};
    auto result = multiply(matrix, scalar);

    REQUIRE(result == expected);
}

TEST_CASE("add: MatrixF[2, 3] x MatrixF[2, 3]")
{
    MatrixF t1 = {{1, 1, 1},
                 {1, 1, 1}};
    MatrixF t2 = {{1, 1, 1},
                 {1, 1, 1}};
    MatrixF expected = {{2, 2, 2},
                       {2, 2, 2}};
    auto result = ts::add(t1, t2);

    REQUIRE(result == expected);
}

TEST_CASE("add(MatrixF, VectorF, axis=0)")
{
    MatrixF matrix = {{1, 1, 1},
                     {0, 0, 0}};
    VectorF vector = {3, 3, 3};
    MatrixF expected = {{4, 4, 4},
                       {3, 3, 3}};
    auto result = ts::add(matrix, vector);

    REQUIRE(result == expected);
}

TEST_CASE("add(Tensor<float, 3>, VectorF)")
{
    Tensor<float, 3> tensor = {{{0, 0}, {1, 1}, {2, 2}},
                                {{3, 3}, {4, 4}, {5, 5}}};
    VectorF vector = {-5, -5};

    Tensor<float, 3> expected = {{{-5, -5}, {-4, -4}, {-3, -3}},
                                 {{-2, -2}, {-1, -1}, {0, 0}}};
    auto result = ts::add(tensor, vector);

    REQUIRE(result == expected);
}

TEST_CASE("divide(MatrixF, VectorF, axis=0)")
{
    MatrixF matrix = {{3, 6, 9},
                     {3, 6, 9}};
    VectorF vector = {3, 3};
    MatrixF expected = {{1, 2, 3},
                       {1, 2, 3}};
    auto result = ts::divide(matrix, vector);

    REQUIRE(result == expected);
}

TEST_CASE("transpose")
{
    MatrixF matrix = {{1, 1, 1},
                     {1, 1, 1}};

    MatrixF expected = {{1, 1},
                       {1, 1},
                       {1, 1}};
    auto result = ts::transpose(matrix);

    REQUIRE(result == expected);
}

TEST_CASE("maximum(scalar, MatrixF)")
{
    MatrixF matrix = {{1, -1, 1},
                     {1, -1, 1}};
    MatrixF expected = {{1, 0, 1},
                       {1, 0, 1}};
    auto result = ts::maximum(0.0f, matrix);

    REQUIRE(result == expected);
}

TEST_CASE("mask from tensor")
{
    MatrixF matrix = {{1, -1, 1},
                     {1, -1, 1}};
    Tensor<char, 2> expected = {{true, false, true},
                                {true, false, true}};

    auto mask = ts::mask<float>(matrix, [](float e) { return e >= 0; });

    REQUIRE(mask == expected);
}

TEST_CASE("assign_if")
{
    MatrixF matrix = {{1, -1, 1},
                     {1, -1, 1}};
    MatrixF expected = {{1, 1337, 1},
                       {1, 1337, 1}};

    auto result = assign_if(matrix, matrix < 0, 1337.0f);

    REQUIRE(result == expected);
}

TEST_CASE("sum(MatrixF, VectorF, axis=0)")
{
    MatrixF matrix = {{1, 1, 1},
                     {1, 1, 1}};
    VectorF expected = {2, 2, 2};

    auto result = ts::sum(matrix, 0);

    REQUIRE(result == expected);
}

TEST_CASE("sum(MatrixF, VectorF, axis=1)")
{
    MatrixF matrix = {{1, 1, 1},
                     {1, 1, 1}};
    VectorF expected = {3, 3};

    auto result = ts::sum(matrix, 1);

    REQUIRE(result == expected);
}

TEST_CASE("sum(Tensor)")
{
    MatrixF matrix = {{1, 1, 1},
                     {1, 1, 1}};
    auto result = ts::sum(matrix);
    REQUIRE(result == 6);
}

TEST_CASE("to_one_hot")
{
    Tensor<int, 1> vector = {2, 0, 1};
    MatrixF matrix = {{1, 2, 3},
                     {1, 2, 3},
                     {1, 2, 3}};

    Tensor<char, 2> expected = {{false, false, true},
                                {true, false, false},
                                {false, true, false}
    };
    Tensor<char, 2> one_hot = ts::to_one_hot(vector);

    REQUIRE(one_hot == expected);
}

TEST_CASE("apply_if")
{
    MatrixF matrix = {{1, -1, 1},
                     {1, -1, 1}};
    MatrixF expected = {{1, 1, 1},
                       {1, 1, 1}};

    auto result = ts::apply_if(matrix, matrix < 0,
                               (Fn<float>)[](float e) { return std::abs(e); });

    REQUIRE(result == expected);
}

TEST_CASE("get(MatrixF, VectorF)")
{
    MatrixF matrix = {{1, 2, 3},
                     {1, 2, 3}};
    Tensor<int, 1> vector = {2, 0};
    VectorF expected = {3, 1};

    auto result = ts::get(matrix, vector);

    REQUIRE(result == expected);
}

TEST_CASE("apply")
{
    MatrixF matrix = {{1, 2, 3},
                     {1, 2, 3}};

    MatrixF expected = {{1, 4, 9},
                       {1, 4, 9}};

    Tensor<float, 2> result = ts::apply<float, 2>(matrix, (Fn<float>)[](float e){ return std::pow(e, 2); });

    REQUIRE(result == expected);
}

TEST_CASE("ts::log")
{
    MatrixF matrix = {{1, 2, 3},
                     {1, 2, 3}};
    Tensor<float, 2> result = ts::log(matrix);
}

TEST_CASE("randint")
{
    auto tensor = ts::randint<2>(0, 2, {300, 3});
    for (auto const &e: tensor) {
        REQUIRE((e >= 0 and e <= 2));
    }
}

TEST_CASE("from_vector")
{
   std::vector<float> std_v = {1,2,3,4,5};
   VectorF vector = ts::from_vector(std_v);

   REQUIRE(std::equal(std_v.begin(), std_v.end(), vector.begin()));
   REQUIRE(vector.shape() == std::array<size_type, 1>{5});
}

TEST_CASE("concatenate vectors with axis = 1")
{
   VectorF v0 = {0, 0, 0};
   VectorF v1 = {1, 1, 1};
   VectorF v2 = {2, 2, 2};
   MatrixF expected = {{0, 1, 2},
                      {0, 1, 2},
                      {0, 1, 2}};

   Tensor<float, 2> result = ts::concatenate<float, 1>({v0, v1, v2});

   REQUIRE(result == expected);
}

TEST_CASE("concatenate vectors with axis = 0")
{
    VectorF v0 = {0, 0, 0};
    VectorF v1 = {1, 1, 1};
    VectorF v2 = {2, 2, 2};
    VectorF expected = {0, 0, 0, 1, 1, 1, 2, 2, 2};

    Tensor<float, 1> result = ts::concatenate<float, 0>({v0, v1, v2});

    REQUIRE(result == expected);
}

TEST_CASE("concatenate vectors with axis = 0 and different sizes")
{
    VectorF v0 = {0, 0, 0};
    VectorF v1 = {1, 1};
    VectorF v2 = {2};
    VectorF expected = {0, 0, 0, 1, 1, 2};

    Tensor<float, 1> result = ts::concatenate<float, 0>({v0, v1, v2});

    REQUIRE(result == expected);
}

TEST_CASE("concatenate matrices")
{
    MatrixF expected = {{0, 0, 0, 0},
                        {-10, 1, 10, 100},
                        {-20, 2, 20, 200},
                        {-30, 3, 30, 300}};
    // axis = 0
    {
        std::vector<MatrixF> matrices =
            {
            {{0, 0, 0, 0},
             {-10, 1, 10, 100}},

            {{-20, 2, 20, 200},
            {-30, 3, 30, 300}}
            };
        MatrixF result = ts::concatenate(matrices, 0);
        REQUIRE(result == expected);
    }

    {
        std::vector<MatrixF> matrices =
            {
                {{0, 0},
                 {-10, 1},
                 {-20, 2},
                 {-30, 3}},
                {{0, 0},
                 {10, 100},
                 {20, 200},
                 {30, 300}},

            };
        MatrixF result = ts::concatenate(matrices, 1);
        REQUIRE(result == expected);
    }
}

TEST_CASE("slice")
{
    MatrixF matrix = {{0, 0, 0},
                      {1, 10, 100},
                      {2, 20, 200}};
    // axis = 0
    {
        MatrixF slice = ts::slice(matrix, 0, 2, 0);
        MatrixF expected = {{0, 0, 0},
                           {1, 10, 100}};
        REQUIRE(slice == expected);
    }

    {
        MatrixF slice = ts::slice(matrix, 1, 3, 0);
        MatrixF expected = {{1, 10, 100},
                            {2, 20, 200}};
        REQUIRE(slice == expected);
    }

    // axis = 1
    {
        MatrixF slice = ts::slice(matrix, 0, 2, 1);
        MatrixF expected = {{0, 0},
                            {1, 10},
                            {2, 20}};
        REQUIRE(slice == expected);
    }

    {
        MatrixF slice = ts::slice(matrix, 1, 3, 1);
        MatrixF expected = {{0, 0},
                            {10, 100},
                            {20, 200}};
        REQUIRE(slice == expected);
    }
}

TEST_CASE("argmax(MatrixF)")
{
    Tensor<float, 2> matrix = {{0, 1, 0},
                               {3, 2, 1},
                               {0, 0, 1}};
    Tensor<int, 1> expected = {1, 0, 2};
    Tensor<int, 1> indexes = ts::argmax(matrix);

    REQUIRE(indexes == expected);
}

TEST_CASE("swap(MatrixF, MatrixF")
{
    MatrixF m1 = {{1, 1}, {2, 2}};
    MatrixF m2 = {{3, 3, 3}, {4, 4,4}};

    auto a = m1.clone();
    auto b = m2.clone();

    ts::swap(a, b);

    REQUIRE(a == m2);
    REQUIRE(b == m1);

}
