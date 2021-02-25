#include "functional_helpers.hpp"

using namespace ts;

auto ts::_get_flatten_tile(Tensor<float, 3> const &image, int size, int row, int col) -> VectorF
{
    MatrixF tile(std::pow(size, 2), image.shape(2));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            VectorF vec = image(i + row, j + col);
            std::copy(vec.begin(), vec.end(), tile(j + i * size).begin());
        }
    }
    return tile.flatten();
}

auto ts::_add_flatten_tile(Tensor<float, 3> &image, Tensor<float, 1> const &tile, int size, int row,
                       int col) -> void
{
    int c = image.shape(2);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < c; ++k) {
                int tile_idx = k + j * c + i * c * size;
                image(i + row, j + col, k) += tile(tile_idx);
            }
        }
    }
}

auto ts::_get_flatten_tile(MatrixF const &image, int size, int row, int col) -> VectorF
{
    VectorF tile(std::pow(size, 2));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            tile(j + i * size) = image(i + row, j + col);
        }
    }
    return tile;
}

auto ts::_calculate_output_dim(int dim_in, int kernel_size, int padding, int stride, int dilatation)
    -> int
{
    return (dim_in + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride + 1;
}
