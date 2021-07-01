#include <tensor/tensor.hpp>

#include "conv_2d.hpp"
#include "conv_2d_helpers.hpp"
#include "im2col.hpp"

auto ts::conv_2d_im2col(ts::Tensor<float, 4> const &images, ts::Tensor<float, 2> const &kernel,
                        ts::Tensor<float, 2> &im2col_buffer, int kernel_size, int stride, int pad, int dilatation)
    -> ts::Tensor<float, 4>
{
    // we assume CHW image format
    ts::size_type batch_size = images.shape(0);
    ts::size_type C_in = images.shape(1);
    ts::size_type H = images.shape(2);
    ts::size_type W = images.shape(3);

    ts::size_type C_out = kernel.shape(0);

    ts::size_type dim_out = ts::_calculate_output_dim(H, kernel_size, pad, stride, dilatation);
    ts::Tensor<float, 3> results(batch_size, C_out, dim_out * dim_out);
    auto const buffer_shape = ts::im2col::im2col_buffer_shape({C_in, H, W}, kernel_size, stride, pad, dilatation);

    //#pragma omp parallel for // TODO: check if using OpenMP here will improve something
    for (int b = 0; b < batch_size; ++b) {
        auto buffer = ts::Tensor<float, 2>(buffer_shape);

        auto image = images(b);
        auto result = results(b);
        ts::im2col::im2col(image, kernel_size, pad, stride, dilatation, buffer);
        ts::dot(kernel, buffer, result, false, false);
    }
    return results.reshape<4>({batch_size, C_out, dim_out, dim_out});
}

auto ts::conv_2d(ts::Tensor<float, 4> const &images, ts::Tensor<float, 2> const &kernel, int kernel_size,
                 size_type stride) -> ts::Tensor<float, 4>
{
    uint batch_size = images.shape(0);
    uint dim_out = ts::_calculate_output_dim(images.shape(1), kernel_size, 0, stride, 1);
    ts::Tensor<float, 4> results(batch_size, dim_out, dim_out, kernel.shape(1));

#pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        auto image = images(b);
        auto result = results(b);
        for (size_type i = 0; i < dim_out; ++i) {
            for (size_type j = 0; j < dim_out; ++j) {
                VectorF tile = _get_flatten_tile(image, kernel_size, i * stride, j * stride);
                VectorF tile_output = ts::dot(kernel, tile, true);
                auto [result_begin, result_end] = result.get_subarray({i, j});
                std::copy(tile_output.begin(), tile_output.end(), result_begin);
            }
        }
    }
    return results;
}

auto ts::conv_2d(ts::Tensor<float, 3> const &image, ts::Tensor<float, 2> const &kernel, int kernel_size,
                 size_type stride) -> ts::Tensor<float, 3>
{
    uint dim_out = ts::_calculate_output_dim(image.shape(0), kernel_size, 0, stride, 1);
    ts::Tensor<float, 3> result(dim_out, dim_out, kernel.shape(1));

    for (size_type i = 0; i < dim_out; ++i) {
        for (size_type j = 0; j < dim_out; ++j) {
            ts::VectorF tile = _get_flatten_tile(image, kernel_size, i * stride, j * stride);
            auto tile_output = ts::dot(kernel, tile, true);
            auto [result_begin, result_end] = result.get_subarray({i, j});
            std::copy(tile_output.begin(), tile_output.end(), result_begin);
        }
    }
    return result;
}

auto ts::conv_2d(ts::MatrixF const &matrix, ts::MatrixF const &kernel, size_type stride) -> ts::MatrixF
{
    assert(kernel.shape(0) == kernel.shape(1));

    auto kernel_flatten = kernel.flatten();
    uint kernel_size = kernel.shape(0);
    uint dim_out = ts::_calculate_output_dim(matrix.shape(0), kernel_size, 0, stride, 1);
    ts::MatrixF result(dim_out, dim_out);

    for (size_type i = 0; i < dim_out; ++i) {
        for (size_type j = 0; j < dim_out; ++j) {
            ts::VectorF tile = _get_flatten_tile(matrix, kernel_size, i * stride, j * stride);
            result(i, j) = ts::dot(kernel_flatten, tile);
        }
    }
    return result;
}

auto ts::conv_2d_backward_im2col(ts::Tensor<float, 4> const &inputs, ts::Tensor<float, 2> const &kernel,
                                 ts::Tensor<float, 2> &im2col_buffer, ts::Tensor<float, 4> const &d_outputs,
                                 int kernel_size, int stride, int pad, int dilatation)
    -> std::tuple<ts::Tensor<float, 4>, ts::Tensor<float, 2>>
{
    uint batch_size = inputs.shape(0);
    uint C_in = inputs.shape(1);
    uint dim_in = inputs.shape(2);

    size_type C_out = d_outputs.shape(1);

    ts::Tensor<float, 4> d_inputs(inputs.shape());
    ts::Tensor<float, 2> d_kernel(kernel.shape());

    size_type dim_out = d_outputs.shape(2);
    auto d_outputs_reshaped = d_outputs.reshape<3>({batch_size, C_out, dim_out * dim_out});

    //#pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        auto d_output = d_outputs_reshaped(b);
        auto input = inputs(b);
        auto d_input = d_inputs(b);

        // backpropagate to input
        ts::dot(kernel, d_output, im2col_buffer, true, false);
        im2col::col2im(im2col_buffer, kernel_size, pad, stride, dilatation, d_input);

        // backpropagate to weight
        im2col::im2col(input, kernel_size, pad, stride, dilatation, im2col_buffer);
        ts::dot(d_output, im2col_buffer, d_kernel, false, true, 1.0f);
    }
    return std::make_tuple(std::move(d_inputs), std::move(d_kernel));
}

auto ts::conv_2d_backward(ts::Tensor<float, 4> const &inputs, ts::Tensor<float, 2> const &kernel,
                          ts::Tensor<float, 4> const &d_outputs, int kernel_size, int stride)
    -> std::tuple<ts::Tensor<float, 4>, ts::Tensor<float, 2>>
{
    uint batch_size = inputs.shape(0);
    uint dim_in = inputs.shape(1);
    uint channels_in = inputs.shape(3);
    uint channels_out = d_outputs.shape(3);

    ts::Tensor<float, 4> d_inputs(batch_size, dim_in, dim_in, channels_in);
    ts::Tensor<float, 2> d_kernel(kernel_size * kernel_size * channels_in, channels_out);

#pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        auto d_output = d_outputs(b);
        auto input = inputs(b);
        auto d_input = d_inputs(b);
        for (int i = 0; i < d_output.shape(0); ++i) {
            for (int j = 0; j < d_output.shape(1); ++j) {
                ts::VectorF d_tile = d_output(i, j);
                ts::VectorF tile = _get_flatten_tile(input, kernel_size, i * stride, j * stride);

                auto d_tile_input = ts::dot(kernel, d_tile);
                _add_flatten_tile(d_input, d_tile_input, kernel_size, i * stride, j * stride);

                auto d_tile_kernel = ts::outer_product(tile, d_tile);
#pragma omp critical
                ts::add_(d_kernel, d_tile_kernel);
            }
        }
    }
    return std::make_tuple(std::move(d_inputs), std::move(d_kernel));
}

auto ts::conv_2d_backward(ts::Tensor<float, 3> const &input, ts::Tensor<float, 2> const &kernel,
                          ts::Tensor<float, 3> const &d_output, int kernel_size, int stride)
    -> std::tuple<ts::Tensor<float, 3>, ts::Tensor<float, 2>>
{
    uint channels_in = input.shape(2);
    uint channels_out = d_output.shape(2);
    uint dim_in = input.shape(1);

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
