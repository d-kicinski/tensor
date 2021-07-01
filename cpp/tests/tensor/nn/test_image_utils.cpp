#include "catch2/catch.hpp"
#include <tensor/nn/im2col.hpp>
#include <tensor/nn/image_utils.hpp>
#include <tensor/tensor.hpp>

TEST_CASE("pad")
{
    ts::MatrixF matrix =
        {
            {1, 1},
            {1, 1}
        };
    {
        auto result = pad(matrix, 1, 0);
        ts::MatrixF expected =
            {
                {0, 0},
                {1, 1},
                {1, 1},
                {0, 0}
            };
        REQUIRE(result == expected);
    }
    {
        auto result = ts::pad(matrix, 0, 1);
        ts::MatrixF expected =
            {
                {0, 1, 1, 0},
                {0, 1, 1, 0}
            };
        REQUIRE(result == expected);
    }
    {
        auto result = pad(matrix, 2, 2);
        ts::MatrixF expected =
            {
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

    ts::Tensor<float, 4> tensor_hwc =
        {{
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
