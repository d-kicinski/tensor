#include <utility>
#include <tensor/tensor.hpp>

#include "conv_2d_helpers.hpp"
#include "max_pool_2d.hpp"


auto find_max(ts::Tensor<float, 3> const &input) -> std::pair<ts::VectorF, std::vector<std::array<int, 3>>>
{
    uint channels = input.shape(2);
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

auto ts::max_pool_2d_hwc(ts::Tensor<float, 4> const &inputs, int kernel_size, int stride)
    -> std::pair<ts::Tensor<float, 4>, ts::Tensor<char, 4>>
{
    int dim_out = ts::_calculate_output_dim(inputs.shape(1), kernel_size, 0, stride, 1);
    int C_in = inputs.shape(3);
    int batch_size = inputs.shape(0);

    ts::Tensor<float, 4> results(batch_size, dim_out, dim_out, C_in);
    ts::Tensor<char, 4> masks(inputs.shape());

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

auto ts::max_pool_2d(ts::Tensor<float, 4> const &inputs, int kernel_size, int stride, int pad)
    -> std::pair<ts::Tensor<float, 4>, ts::Tensor<int, 4>>
{
    int batch_size = inputs.shape(0);
    int C_in = inputs.shape(1);
    int dim_out = ts::_calculate_output_dim(inputs.shape(2), kernel_size, pad, stride, 1);

    int dim_out_h = dim_out;
    int dim_out_w = dim_out;
    int pad_h = pad;
    int pad_w = pad;
    int stride_h = stride;
    int stride_w = stride;
    int kernel_h = kernel_size;
    int kernel_w = kernel_size;

    int height = inputs.shape(2);
    int width = inputs.shape(3);

    ts::Tensor<float, 4> results(batch_size, C_in, dim_out_h, dim_out_w);
    ts::fill_(results, -std::numeric_limits<float>::infinity());
    ts::Tensor<int, 4> masks(batch_size, C_in, dim_out_h, dim_out_w);

#pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        auto input = inputs(b);
        auto mask = masks(b);
        auto result = results(b);
        for (int c = 0; c < C_in; ++c) {
            for (int i = 0; i < dim_out_h; ++i) {
                for (int j = 0; j < dim_out_w; ++j) {

                    // get two corners of current tile
                    int h_start = i * stride_h - pad_h;
                    int w_start = j * stride_w - pad_w;
                    int h_end = std::min(h_start + kernel_h, height);
                    int w_end = std::min(w_start + kernel_w, width);
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);

                    // iterate over a current tile
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            const int index = h * width + w;
                            if (input.at({c, h, w}) > result.at({c, i, j})) {
                                result.at({c, i, j}) = input.at({c, h, w});
                                mask.at({c, i, j}) = input.index({c, h, w});
                            }
                        }
                    }
                }
            }
        }
    }
    return std::make_pair(results, masks);
}

auto put_vector_to_tile(ts::Tensor<float, 3> &d_input_tile, ts::VectorF const &vector, ts::Tensor<char, 3> const &mask)
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

auto ts::max_pool_2d_backward(ts::Tensor<float, 4> const &d_outputs, ts::Tensor<int, 4> const &masks, int dim_in,
                              int kernel_size, int stride) -> ts::Tensor<float, 4>
{
    int batch_size = masks.shape(0);
    int C_in = d_outputs.shape(1);
    int dim_out_h = d_outputs.shape(2);
    int dim_out_w = d_outputs.shape(3);

    auto d_inputs = ts::Tensor<float, 4>(batch_size, C_in, dim_in, dim_in);

#pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        auto mask = masks(b);
        auto d_output = d_outputs(b);
        auto d_input = d_inputs(b);

        for (int c = 0; c < C_in; ++c) {
            for (int i = 0; i < dim_out_h; ++i) {
                for (int j = 0; j < dim_out_w; ++j) {
                    d_input.at(mask.at({c, i, j})) += d_output.at({c, i, j});
                }
            }
        }
    }
    return d_inputs;
}

auto ts::max_pool_2d_backward_hwc(ts::Tensor<float, 4> const &d_outputs, ts::Tensor<char, 4> const &masks,
                                  int kernel_size, int stride) -> ts::Tensor<float, 4>
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
