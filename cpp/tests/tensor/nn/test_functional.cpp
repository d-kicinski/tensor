#include "catch2/catch.hpp"
#include <tensor/tensor.hpp>
#include <tensor/nn/functional.hpp>


TEST_CASE("conv_2d(Tensor<float, 3>, ...")
{
    // shape: (H, W, C_in)
    ts::Tensor<float, 3> input =
        {{{1, -1}, {2, -2}, {3, -3}},
         {{4, -4}, {5, -5}, {6, -6}},
         {{7, -7}, {8, -8}, {9, -9}}};

    // shape: (k*k*C_in, c_out)
    ts::Tensor<float, 2> kernel = {
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
    };

    ts::Tensor<float, 3> expected_output = {
        {{48, 24}, {64, 32}},
        {{96, 48}, {112, 56}}
    };

    auto output = ts::conv_2d(input, kernel, 2, 1);

    REQUIRE(output == expected_output);
}

TEST_CASE("conv_2d(MatrixF, ...)")
{
    ts::MatrixF matrix = {{10, 20, 30, 40},
                          {11, 21, 31, 41},
                          {12, 22, 32, 42},
                          {13, 23, 33, 43}};
    ts::MatrixF kernel = {{2, 2},
                          {2, 2}};

    {
        ts::MatrixF result = ts::conv_2d(matrix, kernel, 1);
        ts::MatrixF expected = {{124, 204, 284},
                                {132, 212, 292},
                                {140, 220, 300}};
        REQUIRE(result.data_size() == 9);
        REQUIRE(result == expected);
    }
    {
        ts::MatrixF result = ts::conv_2d(matrix, kernel, 2);
        ts::MatrixF expected = {{124, 284},
                                {140, 300}};
        REQUIRE(result.data_size() == 4);
        REQUIRE(result == expected);
    }
}

TEST_CASE("conv_2d_gradient")
{
    // shape: (H, W, C_in)
    ts::Tensor<float, 3> input =
        {{{1, -1}, {2, -2}, {3, -3}},
         {{4, -4}, {5, -5}, {6, -6}},
         {{7, -7}, {8, -8}, {9, -9}}};


    // shape: (k*k*C_in, c_out)
    ts::Tensor<float, 2> kernel = {
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
    };

    // shape: (H', W', C_out)
    ts::Tensor<float, 3> d_output = {
        {{1, -1}, {2, -2}},
        {{3, -3}, {4, -4}}
    };

    assert((kernel.shape() == std::array<ts::size_type, 2>{8, 2}));

    int kernel_size = 2;
    int stride = 1;

    auto output = ts::conv_2d(input, kernel, kernel_size, stride);

    auto [d_input, d_kernel] =  ts::conv_2d_backward(input, kernel, d_output, kernel_size, stride);

    ts::Tensor<float, 3> expected_d_input =
        {{{1, -1}, {3, -3}, {2, -2}},
         {{4, -4}, {10, -10}, {6, -6}},
         {{3, -3}, {7, -7}, {4, -4}}};

    ts::Tensor<float, 2> expected_d_kernel = {
        {37, -37},
        {-37, 37},
        {47, -47},
        {-47, 47},
        {67, -67},
        {-67, 67},
        {77, -77},
        {-77, 77},
    };

    REQUIRE(d_input == expected_d_input);
    REQUIRE(d_kernel == expected_d_kernel);
}


TEST_CASE("pad")
{
    ts::MatrixF matrix = {{1, 1},
                      {1, 1}};
    {
        auto result = pad(matrix, 1, 0);
        ts::MatrixF expected = {{0, 0},
                            {1, 1},
                            {1, 1},
                            {0, 0}};
        REQUIRE(result == expected);
    }
    {
        auto result = ts::pad(matrix, 0, 1);
        ts::MatrixF expected = {{0, 1, 1, 0},
                            {0, 1, 1, 0}};
        REQUIRE(result == expected);
    }
    {
        auto result = pad(matrix, 2, 2);
        ts::MatrixF expected = {
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {0, 0, 1, 1, 0, 0},
            {0, 0, 1, 1, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0}
        };
        REQUIRE(result == expected);
    }
}

TEST_CASE("max_pool_2d")
{
   ts::Tensor<float, 4> input = {{
        {{0, 0}, {1, 0}, {1, 0}, {0, 0}},
        {{0, 1}, {0, 0}, {0, 0}, {0, 1}},
        {{0, 1}, {0, 0}, {0, 0}, {0, 1}},
        {{0, 0}, {1, 0}, {1, 0}, {0, 0}}
    }};

    ts::Tensor<float, 4> expected_output = {{
       {{1, 1}, {1, 1}},
       {{1, 1}, {1, 1}},
   }};

    ts::Tensor<bool, 4> expected_mask = {{
        {{false, false}, {true, false}, {true, false}, {false, false}},
        {{false, true}, {false, false}, {false, false}, {false, true}},
        {{false, true}, {false, false}, {false, false}, {false, true}},
        {{false, false}, {true, false}, {true, false}, {false, false}},
    }};

    auto [output, mask] = ts::max_pool_2d(input, 2, 2);

    REQUIRE(output.shape() == std::array<ts::size_type, 4>{1, 2, 2, 2});
    REQUIRE(output == expected_output);
    REQUIRE(mask == expected_mask);

    ts::Tensor<float, 4> d_output = {{
        {{2, 3}, {4, 5}},
        {{6, 7}, {8, 9}},
    }};

    ts::Tensor<float, 4> expected_d_input = {{
        {{0, 0}, {2, 0}, {4, 0}, {0, 0}},
        {{0, 3}, {0, 0}, {0, 0}, {0, 5}},
        {{0, 7}, {0, 0}, {0, 0}, {0, 9}},
        {{0, 0}, {6, 0}, {8, 0}, {0, 0}}
    }};

    auto d_input = ts::max_pool_2d_backward(d_output, mask, 2, 2);
    REQUIRE(d_input.shape() == expected_d_input.shape());
    REQUIRE(d_input == expected_d_input);
}

