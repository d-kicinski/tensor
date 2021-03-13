#include "functional_helpers.hpp"

using namespace ts;

template auto ts::_get_tile(Tensor<float, 3> const &image, int size, int row, int col) -> Tensor<float, 3>;
template auto ts::_get_tile(Tensor<bool, 3> const &image, int size, int row, int col) -> Tensor<bool, 3>;

template auto ts::_set_tile(Tensor<float, 3> &image, Tensor<float, 3> const &tile, int size, int row, int col) -> void;
template auto ts::_set_tile(Tensor<bool, 3> &image, Tensor<bool, 3> const &tile, int size, int row, int col) -> void;

auto ts::_get_flatten_tile(Tensor<float, 4> const &images, int size, int row, int col) -> MatrixF
{
    std::vector<VectorF> tiles;
    for (int b = 0; b < images.shape(0); ++b) {
        auto image = images(b);
        MatrixF tile(std::pow(size, 2), image.shape(2));
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                VectorF vec = image(i + row, j + col);
                std::copy(vec.begin(), vec.end(), tile(j + i * size).begin());
            }
        }
        tiles.push_back(std::move(tile.flatten()));
    }
    return MatrixF(tiles);  // (B, k*k*C_in)
}

auto ts::_get_flatten_tile(Tensor<float, 3> const &image, int size, int row, int col) -> VectorF
{
    return _get_tile(image, size, row, col).flatten();
}

auto ts::_add_flatten_tile(Tensor<float, 4> &images, Tensor<float, 2> const &tiles, int size, int row,
                           int col) -> void
{
    for (int b = 0; b < images.shape(0); ++b) {
        auto image = images(b);
        auto tile = tiles(b);
        int c = image.shape(3);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                for (int k = 0; k < c; ++k) {
                    int tile_idx = k + j * c + i * c * size;
                    image(i + row, j + col, k) += tile(tile_idx);
                }
            }
        }
    }
}

auto ts::_add_flatten_tile(Tensor<float, 3> &image, Tensor<float, 1> const &tile, int size, int row,
                       int col) -> void
{
    int c = image.shape(2);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < c; ++k) {
                int tile_idx = k + j * c + i * c * size;
                image.at({i + row, j + col, k}) += tile(tile_idx);
            }
        }
    }
}

template<typename Element>
auto ts::_get_tile(Tensor<Element, 3> const &image, int size, int row, int col) -> Tensor<Element, 3>
{
    Tensor<Element, 3> tile(size, size, image.shape(2));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            auto[vec_begin, vec_end] = image.get_subarray({i + row, j + col});
            auto[tile_begin, tile_end] = tile.get_subarray({i , j});
            std::copy(vec_begin, vec_end, tile_begin);
        }
    }
    return tile;
}

template <typename Element>
auto ts::_set_tile(Tensor<Element, 3> &image, Tensor<Element, 3> const &tile, int size, int row,
                           int col) -> void
{
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            auto vec = tile(i, j);
            std::copy(vec.begin(), vec.end(), image(i + row, j + col).begin());
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
