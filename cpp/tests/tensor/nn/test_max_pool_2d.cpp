#include <catch2/catch.hpp>

#include <tensor/nn/image_utils.hpp>
#include <tensor/nn/initialization.hpp>
#include <tensor/nn/max_pool_2d.hpp>
#include <tensor/tensor.hpp>

TEST_CASE("max_pool_2d_hwc")
{
    ts::Tensor<float, 4> input =
        {{
            {{0, 0}, {1, 0}, {1, 0}, {0, 0}},
            {{0, 1}, {0, 0}, {0, 0}, {0, 1}},
            {{0, 1}, {0, 0}, {0, 0}, {0, 1}},
            {{0, 0}, {1, 0}, {1, 0}, {0, 0}}
        }};

    ts::Tensor<float, 4> expected_output =
        {{
            {{1, 1}, {1, 1}},
            {{1, 1}, {1, 1}},
        }};

    ts::Tensor<char, 4> expected_mask =
        {{
            {{false, false}, {true, false}, {true, false}, {false, false}},
            {{false, true}, {false, false}, {false, false}, {false, true}},
            {{false, true}, {false, false}, {false, false}, {false, true}},
            {{false, false}, {true, false}, {true, false}, {false, false}},
        }};

    auto [output, mask] = ts::max_pool_2d_hwc(input, 2, 2);

    REQUIRE(output.shape() == std::array<ts::size_type, 4>{1, 2, 2, 2});
    REQUIRE(output == expected_output);
    REQUIRE(mask == expected_mask);

    ts::Tensor<float, 4> d_output =
        {{
            {{2, 3}, {4, 5}},
            {{6, 7}, {8, 9}},
        }};

    ts::Tensor<float, 4> expected_d_input =
        {{
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
