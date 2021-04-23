#include "functional.hpp"
#include "functional_helpers.hpp"

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

auto ts::conv_2d_im2col(ts::Tensor<float, 4> const &images, ts::Tensor<float, 2> const &kernel,
                        ts::Tensor<float, 2> &im2col_buffer, int kernel_size, int stride, int pad, int dilatation)
    -> ts::Tensor<float, 4>
{
    ts::size_type batch_size = images.shape(0);

    // we assume CHW image format
    ts::size_type C_in = images.shape(1);
    ts::size_type H = images.shape(2);
    ts::size_type W = images.shape(3);

    ts::size_type C_out = kernel.shape(0);

    ts::size_type dim_out = ts::_calculate_output_dim(H, kernel_size, pad, stride, dilatation);
    ts::Tensor<float, 4> results(batch_size, C_out, dim_out, dim_out);

#pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        auto image = images(b);
        auto result = results(b);
        im2col::im2col(image.raw_data(), C_in, H, W, kernel_size, kernel_size, pad, pad, stride, stride, dilatation, dilatation,
               im2col_buffer.raw_data_mutable());
        auto result_reshaped = result.reshape<2>({C_out, dim_out * dim_out});
        ts::dot(kernel, im2col_buffer, result_reshaped, false, false);
        result = result_reshaped.reshape<3>({C_out, dim_out, dim_out});
    }
    return results;
}

auto ts::conv_2d(ts::Tensor<float, 4> const &images, ts::Tensor<float, 2> const &kernel, int kernel_size,
                 size_type stride) -> ts::Tensor<float, 4>
{
    int batch_size = images.shape(0);
    int dim_out = ts::_calculate_output_dim(images.shape(1), kernel_size, 0, stride, 1);
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
    int dim_out = ts::_calculate_output_dim(image.shape(0), kernel_size, 0, stride, 1);
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
    int kernel_size = kernel.shape(0);
    int dim_out = ts::_calculate_output_dim(matrix.shape(0), kernel_size, 0, stride, 1);
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
    int batch_size = inputs.shape(0);
    int dim_in = inputs.shape(2);
    int channels_in = inputs.shape(1);
    int channels_out = d_outputs.shape(1);

    ts::Tensor<float, 4> d_inputs(batch_size, channels_in, dim_in, dim_in);
    ts::Tensor<float, 2> d_kernel(channels_out, kernel_size * kernel_size * channels_in);

    size_type C = d_outputs.shape(1);
    size_type dim_out = d_outputs.shape(2);

#pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        auto d_output = d_outputs(b);
        auto input = inputs(b);
        auto d_input = d_inputs(b);

        // reshape d_output for dot operation
        auto d_output_reshaped = d_output.reshape<2>({C, dim_out * dim_out});

        // backpropagate to input
        ts::dot(kernel, d_output_reshaped, im2col_buffer, false, false);
        im2col::col2im(im2col_buffer.raw_data(), C, dim_in, dim_in, kernel_size, kernel_size, pad, pad, stride, stride,
               dilatation, dilatation, d_input.raw_data_mutable());


        // backpropagate to weight
        im2col::col2im(input.raw_data(), C, dim_in, dim_in, kernel_size, kernel_size, pad, pad, stride, stride,
               dilatation, dilatation, im2col_buffer.raw_data_mutable());
        ts::dot(d_output_reshaped, im2col_buffer, d_kernel, false, true);

        // restore initial shape
        d_output = d_output_reshaped.reshape<3>({C, dim_out, dim_out});

    }
    return std::make_tuple(std::move(d_inputs), std::move(d_kernel));
}

auto ts::conv_2d_backward(ts::Tensor<float, 4> const &inputs, ts::Tensor<float, 2> const &kernel,
                          ts::Tensor<float, 4> const &d_outputs, int kernel_size, int stride)
    -> std::tuple<ts::Tensor<float, 4>, ts::Tensor<float, 2>>
{
    int batch_size = inputs.shape(0);
    int dim_in = inputs.shape(1);
    int channels_in = inputs.shape(3);
    int channels_out = d_outputs.shape(3);

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

auto find_max(ts::Tensor<float, 3> const &input) -> std::pair<ts::VectorF, std::vector<std::array<int, 3>>>
{
    int channels = input.shape(2);
    auto max_values = ts::VectorF(channels);
    auto indices = std::vector<std::array<int, 3>>(channels);

    for (auto &v : max_values) {
        v = -std::numeric_limits<float>::infinity();
    }

    for (int i = 0; i < input.shape(0); ++i) {
        for (int j = 0; j < input.shape(1); ++j) {
            for (int c = 0; c < channels; ++c) {
                auto value = input(i, j, c);
                if (value > max_values[c]) {
                    max_values[c] = value;
                    indices.at(c) = {i, j, c};
                }
            }
        }
    }
    return std::make_pair(max_values, indices);
}

auto ts::max_pool_2d(ts::Tensor<float, 4> const &inputs, int kernel_size, int stride)
    -> std::pair<ts::Tensor<float, 4>, ts::Tensor<bool, 4>>
{
    int dim_out = ts::_calculate_output_dim(inputs.shape(1), kernel_size, 0, stride, 1);
    int C_in = inputs.shape(3);
    int batch_size = inputs.shape(0);

    ts::Tensor<float, 4> results(batch_size, dim_out, dim_out, C_in);
    ts::Tensor<bool, 4> masks(inputs.shape());

#pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        auto input = inputs(b);
        auto mask = masks(b);
        auto result = results(b);
        for (int i = 0; i < dim_out; ++i) {
            for (int j = 0; j < dim_out; ++j) {
                auto tile = _get_tile(input, kernel_size, i * stride, j * stride);
                auto [max_values, indices] = find_max(tile);
                for (auto &idx : indices) {
                    mask[i * stride + idx[0]][j * stride + idx[1]][idx[2]] = true;
                }

                // TODO copy seems unnecessary
                //   something like this might be cool: image(i, j) = std::move(max_values);
                std::copy(max_values.begin(), max_values.end(), result(i, j).begin());
            }
        }
    }
    return std::make_pair(results, masks);
}

auto put_vector_to_tile(ts::Tensor<float, 3> &d_input_tile, ts::VectorF const &vector, ts::Tensor<bool, 3> const &mask)
    -> void
{
    for (int i = 0; i < d_input_tile.shape(0); ++i) {
        for (int j = 0; j < d_input_tile.shape(1); ++j) {
            for (int k = 0; k < d_input_tile.shape(2); ++k) {
                if (mask(i, j, k)) {
                    d_input_tile(i, j, k) = vector(k);
                }
            }
        }
    }
}

auto ts::max_pool_2d_backward(ts::Tensor<float, 4> const &d_outputs, ts::Tensor<bool, 4> const &masks, int kernel_size,
                              int stride) -> ts::Tensor<float, 4>
{
    auto d_inputs = ts::Tensor<float, 4>(masks.shape());
    int dim_out = d_outputs.shape(1);
    int batch_size = masks.shape(0);

#pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        auto mask = masks(b);
        auto d_output = d_outputs(b);
        auto d_input = d_inputs(b);

        for (int i = 0; i < dim_out; ++i) {
            for (int j = 0; j < dim_out; ++j) {
                auto mask_tile = _get_tile(mask, kernel_size, i * stride, j * stride);
                auto d_input_tile = ts::Tensor<float, 3>(mask_tile.shape());
                put_vector_to_tile(d_input_tile, d_output(i, j), mask_tile);
                _set_tile(d_input, d_input_tile, kernel_size, i * stride, j * stride);
            }
        }
    }
    return d_inputs;
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
