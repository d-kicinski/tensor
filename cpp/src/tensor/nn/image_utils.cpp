#include "image_utils.hpp"

#include "im2col.hpp"

auto ts::pad(ts::MatrixF const &matrix, int pad_row, int pad_col) -> ts::MatrixF
{
    ts::MatrixF matrix_padded(matrix.shape(0) + 2 * pad_row, matrix.shape(1) + 2 * pad_col);
    for (int i = 0; i < matrix.shape(0); ++i) {
        for (int j = 0; j < matrix.shape(1); ++j) {
            matrix_padded(i + pad_row, j + pad_col) = matrix(i, j);
        }
    }
    return matrix_padded;
}

auto ts::hwc2chw(ts::Tensor<float, 4> const &hwc_tensor) -> ts::Tensor<float, 4>
{
    ts::size_type B = hwc_tensor.shape(0);
    ts::size_type H = hwc_tensor.shape(1);
    ts::size_type W = hwc_tensor.shape(2);
    ts::size_type C = hwc_tensor.shape(3);
    ts::Tensor<float, 4> result(std::array<ts::size_type, 4>{B, C, H, W});

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    result.at({b, c, h, w}) = hwc_tensor.at({b, h, w, c});
                }
            }
        }
    }
    return result;
}
auto ts::chw2hwc(ts::Tensor<float, 4> const &chw_tensor) -> ts::Tensor<float, 4>
{
    ts::size_type B = chw_tensor.shape(0);
    ts::size_type C = chw_tensor.shape(1);
    ts::size_type H = chw_tensor.shape(2);
    ts::size_type W = chw_tensor.shape(3);
    ts::Tensor<float, 4> result(std::array<ts::size_type, 4>{B, H, W, C});

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    result.at({b, h, w, c}) = chw_tensor.at({b, c, h, w});
                }
            }
        }
    }
    return result;
}
