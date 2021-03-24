#pragma once
#include "tensor/tensor.hpp"

namespace ts {

auto _get_flatten_tile(Tensor<float, 4> const &image, int size, int row, int col) -> MatrixF;

auto _get_flatten_tile(Tensor<float, 3> const &image, int size, int row, int col) -> VectorF;

auto _add_flatten_tile(Tensor<float, 4> &images, Tensor<float, 2> const &tiles, int size,
                       int row, int col) -> void;


auto _add_flatten_tile(Tensor<float, 3> &image, Tensor<float, 1> const &tile, int size,
                       int row, int col) -> void;

auto _get_flatten_tile(MatrixF const &image, int size, int row, int col) -> VectorF;

auto _calculate_output_dim(int dim_in, int kernel_size, int padding, int stride, int dilatation)
    -> int;

template <typename Element>
auto _get_tile(Tensor<Element, 3> const &image, size_type size, size_type row, size_type col) -> Tensor<Element, 3>;

template <typename Element>
auto _set_tile(Tensor<Element, 3> &image, Tensor<Element, 3> const &tile, int size, int row,
               int col) -> void;

} // namespace ts
