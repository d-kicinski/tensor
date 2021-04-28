#include "catch2/catch.hpp"
#include <tensor/nn/functional.hpp>
#include <tensor/nn/im2col.hpp>
#include <tensor/tensor.hpp>
#include <tensor/nn/initialization.hpp>

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

TEST_CASE("conv_2d_im2col(Tensor<float, 3>, ...")
{
    // shape: (C_int, H, W)
    ts::Tensor<float, 4> input =
        {{
            {{1, 2, 3},
             {4, 5, 6},
             {7, 8, 9}},
            {{-1, -2, -3},
             {-4, -5, -6},
             {-7, -8, -9}},
        }};

    // shape: (c_out, k*k*C_in, )
    ts::Tensor<float, 2> kernel = {
        {2, 2, 2, 2, -2, -2, -2, -2},
        {1, 1, 1, 1, -1, -1, -1, -1}
    };

    ts::Tensor<float, 4> expected_output = {{
        {{48, 64},
         {96, 112}},
        {{24, 32},
         {48, 56}}
    }};


    auto const im2col_buffer_shape = ts::im2col::im2col_buffer_shape({2, 3, 3},
                                                                 2, 1, 0, 1);
    auto _im2col_buffer = ts::Tensor<float, 2>(im2col_buffer_shape);

    auto output = ts::conv_2d_im2col(input, kernel, _im2col_buffer, 2, 1, 0, 1);

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

TEST_CASE("max_pool_2d_hwc")
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

    auto [output, mask] = ts::max_pool_2d_hwc(input, 2, 2);

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

    auto d_input = ts::max_pool_2d_backward_hwc(d_output, mask, 2, 2);
    REQUIRE(d_input.shape() == expected_d_input.shape());
    REQUIRE(d_input == expected_d_input);
}

TEST_CASE("max_pool_2d")
{
    ts::Tensor<float, 4> input =
        {{
            {{0, 1, 1, 0},
             {0, 0, 0, 0},
             {0, 0, 0 , 0},
             {0, 1, 1, 0}},
            {{0, 0, 0 , 0},
             {1, 0 , 0, 1},
             {1, 0, 0, 1},
             {0, 0 , 0, 0}}
        }};

    ts::Tensor<float, 4> expected_output =
        {{
            {{1, 1},
             {1, 1}},
            {{1, 1},
             {1, 1}}
        }};

    ts::Tensor<int, 4> expected_mask =
        {{
            {{1, 2},
             {13, 14}},
            {{20, 23},
             {24, 27}}
        }};

    auto [output, mask] = ts::max_pool_2d(input, 2, 2, 0);

    REQUIRE(output.shape() == std::array<ts::size_type, 4>{1, 2, 2, 2});
    REQUIRE(output == expected_output);
    REQUIRE(mask == expected_mask);

    ts::Tensor<float, 4> d_output =
        {{
            {{2, 4},
             {6, 8}},
            {{3, 5},
             {7, 9}}
        }};

    ts::Tensor<float, 4> expected_d_input =
        {{
             {{0, 2, 4, 0},
              {0, 0, 0, 0},
              {0, 0, 0 , 0},
              {0, 6, 8, 0}},
             {{0, 0, 0 , 0},
              {3, 0 , 0, 5},
              {7, 0, 0, 9},
              {0, 0 , 0, 0}}
         }};

    int dim_in = input.shape(2);
    auto d_input = ts::max_pool_2d_backward(d_output, mask, dim_in, 2, 2);
    REQUIRE(d_input.shape() == expected_d_input.shape());
    REQUIRE(d_input == expected_d_input);
}

TEST_CASE("chw2hwc/hwc2chw") {

    ts::Tensor<float, 4> tensor_chw =
        {{
             {{0, 1, 1, 0},
              {0, 0, 0, 0},
              {0, 0, 0 , 0},
              {0, 1, 1, 0}},
             {{0, 0, 0 , 0},
              {1, 0 , 0, 1},
              {1, 0, 0, 1},
              {0, 0 , 0, 0}}
         }};

    ts::Tensor<float, 4> tensor_hwc = {{
                                      {{0, 0}, {1, 0}, {1, 0}, {0, 0}},
                                      {{0, 1}, {0, 0}, {0, 0}, {0, 1}},
                                      {{0, 1}, {0, 0}, {0, 0}, {0, 1}},
                                      {{0, 0}, {1, 0}, {1, 0}, {0, 0}}
                                  }};

    auto tensor_hwc_probably = ts::chw2hwc(tensor_chw);
    REQUIRE(tensor_hwc_probably.shape() == std::array<ts::size_type, 4>{1, 4, 4, 2});
    REQUIRE(tensor_hwc_probably == tensor_hwc);

    auto tensor_chw_probably = ts::hwc2chw(tensor_hwc);
    REQUIRE(tensor_chw_probably.shape() == std::array<ts::size_type, 4>{1, 2, 4, 4});
    REQUIRE(tensor_chw_probably == tensor_chw);
}

TEST_CASE("max_pool_regression_test") {
    ts::size_type B = 1;
    ts::size_type H = 16;
    ts::size_type W = 16;
    ts::size_type C_in = 2;
    ts::size_type K = 2;
    ts::size_type S = 2;
    ts::size_type P = 0;


    ts::Tensor<float, 4> input_hwc = ts::kaiming_uniform<float, 4>({(int)B, (int)H, (int)W, (int)C_in});

    auto [output_hwc, mask_hwc] = ts::max_pool_2d_hwc(input_hwc, K, S);
    auto [output, mask] = ts::max_pool_2d(ts::hwc2chw(input_hwc), K, S, P);
    REQUIRE(output_hwc.shape() == ts::chw2hwc(output).shape());
    REQUIRE(output_hwc == ts::chw2hwc(output));

    ts::size_type H_out = output_hwc.shape(1);
    ts::size_type W_out = output_hwc.shape(2);
    ts::Tensor<float, 4> d_output_hwc = ts::kaiming_uniform<float, 4>({(int)B, (int)H_out, (int)W_out, (int)C_in});

    auto d_input_hwc = ts::max_pool_2d_backward_hwc(d_output_hwc, mask_hwc, K, S);
    auto d_input = ts::max_pool_2d_backward(ts::hwc2chw(d_output_hwc), mask, H, K, S);
    REQUIRE(d_input_hwc.shape() == ts::chw2hwc(d_input).shape());
    REQUIRE(d_input_hwc == ts::chw2hwc(d_input));
}

template <typename AnyTensor>
auto assert_almost_equal(AnyTensor const &t1, AnyTensor const &t2) -> void {
    for (int i = 0; i < t1.data_size(); ++i) {
        REQUIRE(t1.at(i) == Approx(t2.at(i)).margin(0.00001f));
    }
}

auto naive_kernel_to_im2col(ts::Tensor<float, 2> const &kernel, int kernel_size, int C_in, int C_out) -> ts::Tensor<float, 2> {
    ts::Tensor<float, 2> result(kernel.shape(1), kernel.shape(0));
    int K = kernel_size * kernel_size;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int i = 0; i < K; ++i) {
            int idx = i * C_in + c_in;
            int idx_res = i + c_in * K;
            for (int c_out = 0; c_out < C_out; ++c_out) {
                result.at({c_out, idx_res}) = kernel.at({idx, c_out});
            }
        }
    }

    return result;
}

TEST_CASE("naive_kernel_to_im2col") {
    ts::Tensor<float, 2> kernel_naive = {
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
    };


    ts::Tensor<float, 2> kernel_im2col_expected = {
        {2, 2, 2, 2, -2, -2, -2, -2},
        {1, 1, 1, 1, -1, -1, -1, -1}
    };

    auto kernel_im2col = naive_kernel_to_im2col(kernel_naive, 2, 2, 2);

    REQUIRE(kernel_im2col == kernel_im2col_expected);
}

TEST_CASE("conv2d_im2col regression test - sanity check")
{
    int K = 2;
    ts::size_type H = 3;
    ts::size_type W = 3;
    ts::size_type C_in = 2;
    int C_out = 2;

    ts::Tensor<float, 4> input_hwc =
        {{{{1, -1}, {2, -2}, {3, -3}},
         {{4, -4}, {5, -5}, {6, -6}},
         {{7, -7}, {8, -8}, {9, -9}}}};


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

    ts::Tensor<float, 4> expected_output_chw =
        {{
        {{48, 64},
         {96, 112}},
        {{24, 32},
         {48, 56}}
    }};

    ts::Tensor<float, 4> expected_output_hwc = {{
        {{48, 24}, {64, 32}},
        {{96, 48}, {112, 56}}
    }};

    auto output_hwc = ts::conv_2d(input_hwc, kernel, K, 1);

    auto const im2col_buffer_shape = ts::im2col::im2col_buffer_shape({C_in, H, W}, K, 1, 0, 1);
    auto _im2col_buffer = ts::Tensor<float, 2>(im2col_buffer_shape);
    auto kernel_T = naive_kernel_to_im2col(kernel, K, C_in, C_out);
    auto output_chw = ts::conv_2d_im2col(ts::hwc2chw(input_hwc), kernel_T, _im2col_buffer, K, 1, 0, 1);

    REQUIRE(expected_output_hwc == ts::chw2hwc(expected_output_chw));
    REQUIRE(output_hwc == expected_output_hwc);
    REQUIRE(output_chw == expected_output_chw);
    REQUIRE(output_hwc == ts::chw2hwc(output_chw));
}

TEST_CASE("conv2d_im2col regression test - random inputs") {
    ts::size_type B = 8;
    ts::size_type H = 16;
    ts::size_type W = 16;
    ts::size_type C_in = 3;
    ts::size_type C_out = 9;
    ts::size_type K = 3;

    ts::Tensor<float, 4> input_hwc = ts::kaiming_uniform<float, 4>({(int)B, (int)H, (int)W, (int)C_in});

    // shape: (k*k*C_in, c_out)
    ts::Tensor<float, 2> kernel_hwc = ts::kaiming_uniform<float, 2>({static_cast<int>(K*K*C_in), (int)C_out});


    auto output_hwc = ts::conv_2d(input_hwc, kernel_hwc, K, 1);

    auto const im2col_buffer_shape = ts::im2col::im2col_buffer_shape({C_in, H, W}, K, 1, 0, 1);
    auto _im2col_buffer = ts::Tensor<float, 2>(im2col_buffer_shape);
    auto kernel_chw = naive_kernel_to_im2col(kernel_hwc, K, C_in, C_out);
    auto input_chw = ts::hwc2chw(input_hwc);
    auto output_chw = ts::conv_2d_im2col(input_chw, kernel_chw, _im2col_buffer, K, 1, 0, 1);
    auto output_chw_hwd = ts::chw2hwc(output_chw);

    REQUIRE(output_hwc.shape() == ts::chw2hwc(output_chw).shape());
    assert_almost_equal(output_hwc, ts::chw2hwc(output_chw));


    // shape: (H', W', C_out)
    ts::size_type H_out = output_hwc.shape(1);
    ts::size_type W_out = output_hwc.shape(2);
    ts::Tensor<float, 4> d_output_hwc = ts::kaiming_uniform<float, 4>({(int)B, (int)H_out, (int)W_out, (int)C_out});
    ts::Tensor<float, 4> d_output_chw = ts::hwc2chw(d_output_hwc);

    auto [d_input_hwc, d_kernel_hwc] =  ts::conv_2d_backward(input_hwc, kernel_hwc , d_output_hwc, K, 1);

    auto [d_input_chw, d_kernel_chw] =  ts::conv_2d_backward_im2col(input_chw, kernel_chw, _im2col_buffer, d_output_chw, K, 1, 0, 1);


    assert_almost_equal(d_input_hwc, ts::chw2hwc(d_input_chw));
    assert_almost_equal(naive_kernel_to_im2col(d_kernel_hwc, K, C_in, C_out), d_kernel_chw);
}