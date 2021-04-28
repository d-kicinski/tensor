#include <tensor/tensor.hpp>

#include "im2col.hpp"

inline bool is_a_ge_zero_and_a_lt_b(long a, long b)
{
    bool value = false;
    if (a >= 0 && a < b)
        value = true;
    return value;
}

void ts::im2col::im2col(ts::Tensor<float, 3> &image, int kernel, int pad, int stride, int dilation,
                        ts::Tensor<float, 2> &buffer)
{
    ts::size_type const channels = image.shape(0);
    ts::size_type const height = image.shape(1);
    ts::size_type const width = image.shape(2);

    ts::size_type const dim_out = (height + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
    ts::size_type const channel_size = height * width;

    float *data_col = buffer.raw_data_mutable();

    for (int c = 0; c < channels; ++c) {
        auto data = image(c);

        for (int kernel_row = 0; kernel_row < kernel; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel; kernel_col++) {

                int input_row = -pad + kernel_row * dilation; // start row of tile

                for (int output_rows = (int)dim_out; output_rows; output_rows--) {

                    if (!is_a_ge_zero_and_a_lt_b(input_row, (int) height)) {
                        // this sets padding in output array
                        for (int output_cols = (int)dim_out; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad + kernel_col * dilation; // start column of tile

                        for (int output_col = (int)dim_out; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, (int)width)) {
                                *(data_col++) = data.at({input_row, input_col});
                            } else {
                                // this also sets padding
                                *(data_col++) = 0;
                            }
                            input_col += stride;
                        }
                    }
                    input_row += stride;
                }
            }
        }
    }
}


void ts::im2col::col2im(ts::Tensor<float, 2> &buffer, int kernel, int pad, int stride, int dilation,
                        ts::Tensor<float, 3> &image)
{
    ts::size_type const channels = image.shape(0);
    ts::size_type const height = image.shape(1);
    ts::size_type const width = image.shape(2);

    ts::fill_(image, float(0));

    ts::size_type const dim_out = (height + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
    ts::size_type const channel_size = height * width;

    float *data_col = buffer.raw_data_mutable();

    for (int c = 0; c < channels; ++c) {
        auto data = image(c);

        for (int kernel_row = 0; kernel_row < kernel; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel; kernel_col++) {
                int input_row = -pad + kernel_row * dilation;
                for (int output_rows = (int)dim_out; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, (int)height)) {
                        data_col += dim_out;
                    } else {
                        int input_col = -pad + kernel_col * dilation;
                        for (int output_col = (int)dim_out; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, (int)width)) {
                                data.at({input_row,  input_col}) += *data_col;
                            }
                            data_col++;
                            input_col += stride;
                        }
                    }
                    input_row += stride;
                }
            }
        }
    }
}

auto ts::im2col::im2col_buffer_shape(const std::array<size_type, 3> &input_shape, int kernel_size, int stride, int pad,
                                     int dilatation) -> std::array<ts::size_type, 2>
{
    // col_buffer_shape: c_in * k * k, out_1 * out_2
    std::array<size_type, 2> output_shape{static_cast<size_type>(input_shape[0] * kernel_size * kernel_size), 1};

    int const num_spatial_axes_ = 2;

    for (int i = 0; i < num_spatial_axes_; ++i) {
        size_type const input_dim = input_shape[i + 1];
        int const kernel_extent = dilatation * (kernel_size - 1) + 1;
        size_type const output_dim = (input_dim + 2 * pad - kernel_extent) / stride + 1;
        output_shape[1] *= output_dim;
    }
    return output_shape;
}
