#pragma once

template <typename Dtype>
void im2col_cpu(const Dtype *data_im, int channels, int height, int width, int kernel_h,
                int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                int dilation_w, Dtype *data_col);

template <typename Dtype>
void col2im_cpu(const Dtype *data_col, int channels, int height, int width, int kernel_h,
                int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                int dilation_w, Dtype *data_im);
