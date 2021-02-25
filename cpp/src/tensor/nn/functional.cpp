#include "functional.hpp"
#include "functional_helpers.hpp"

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

auto ts::conv_2d(ts::Tensor<float, 3> const &image, ts::Tensor<float, 2> const &kernel,
                 int kernel_size, int stride) -> ts::Tensor<float, 3>
{
    int dim_out = ts::_calculate_output_dim(image.shape(0), kernel_size, 0, stride, 1);
    ts::Tensor<float, 3> result(dim_out, dim_out, kernel.shape(1));

    for (int i = 0; i < dim_out; ++i) {
        for (int j = 0; j < dim_out; ++j) {
            ts::VectorF tile = _get_flatten_tile(image, kernel_size, i * stride, j * stride);
            result(i, j) = ts::dot(ts::transpose(kernel), tile);
        }
    }
    return result;
}

auto ts::conv_2d(ts::MatrixF const &matrix, ts::MatrixF const &kernel, int stride) -> ts::MatrixF
{
    assert(kernel.shape(0) == kernel.shape(1));

    auto kernel_flatten = kernel.flatten();
    int kernel_size = kernel.shape(0);
    int dim_out = ts::_calculate_output_dim(matrix.shape(0), kernel_size, 0, stride, 1);
    ts::MatrixF result(dim_out, dim_out);

    for (int i = 0; i < dim_out; ++i) {
        for (int j = 0; j < dim_out; ++j) {
            ts::VectorF tile = _get_flatten_tile(matrix, kernel_size, i * stride, j * stride);
            result(i, j) = ts::dot(kernel_flatten, tile);
        }
    }
    return result;
}

auto ts::conv_2d_backward(ts::Tensor<float, 3> const &input, ts::Tensor<float, 2> const &kernel,
                          ts::Tensor<float, 3> const &d_output, int kernel_size, int stride)
    -> std::tuple<ts::Tensor<float, 3>, ts::Tensor<float, 2>>
{
    int channels_in = input.shape(2);
    int channels_out = d_output.shape(2);
    int dim_in = input.shape(1);

    ts::Tensor<float, 3> d_input(dim_in, dim_in, channels_in);
    ts::Tensor<float, 2> d_kernel(kernel_size * kernel_size * channels_in, channels_out);

    for (int i = 0; i < d_output.shape(0); ++i) {
        for (int j = 0; j < d_output.shape(1); ++j) {
            ts::VectorF d_tile = d_output(i, j);
            ts::VectorF tile = _get_flatten_tile(input, kernel_size, i * stride, j * stride);

            auto d_tile_kernel = ts::outer_product(tile, d_tile);
            auto d_tile_input = ts::dot(kernel, d_tile);

            _add_flatten_tile(d_input, d_tile_input, kernel_size, i * stride, j * stride);
            ts::add_(d_kernel, d_tile_kernel);
        }
    }
    return std::make_tuple(std::move(d_input), std::move(d_kernel));
}
