#include <catch2/catch.hpp>

#include <tensor/ts.hpp>

auto get_flatten_tile(ts::Tensor<float, 3> const &image, int size, int row, int col) -> ts::VectorF
{
    ts::MatrixF tile(std::pow(size, 2), image.shape(2));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            tile(j + i * size) = image(i + row , j + col);
        }
    }
    return tile.flatten();
}

auto add_flatten_tile(ts::Tensor<float, 3> &image,
                      ts::Tensor<float, 1> const &tile,
                      int size, int row, int col)
{
    int c = image.shape(2);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < c; ++k) {
                int tile_idx = k + j*c + i*c*size;
                image(i + row , j + col, k)  += tile(tile_idx);
            }
        }
    }
}

auto get_flatten_tile(ts::MatrixF const &image, int size, int row, int col) -> ts::VectorF
{
    ts::VectorF tile(std::pow(size, 2));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            tile(j + i * size) = image(i + row , j + col);
        }
    }
    return tile;
}


auto calculate_output_dim(int dim_in, int kernel_size, int padding = 0, int stride = 1,
                          int dilatation = 1) -> int
{
    return (dim_in + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride + 1;
}

auto conv2d(ts::MatrixF const &matrix, ts::MatrixF kernel, int stride) -> ts::MatrixF
{
    assert(kernel.shape(0) == kernel.shape(1));

    auto kernel_flatten = kernel.flatten();
    int kernel_size = kernel.shape(0);
    int dim_out = calculate_output_dim(matrix.shape(0), kernel_size, 0, stride, 1);
    ts::MatrixF result(dim_out, dim_out);

    for (int i = 0; i < dim_out; ++i) {
        for (int j = 0; j < dim_out; ++j) {
            ts::VectorF tile = get_flatten_tile(matrix, kernel_size, i*stride, j*stride);
            result(i, j) = ts::dot(kernel_flatten, tile);
        }
    }
    return result;
}


auto conv2d(ts::Tensor<float, 3> const & image,
            ts::Tensor<float, 2> const & kernel,
            int kernel_size,
            int stride) -> ts::Tensor<float, 3>
{
    //    auto kernel_flatten = kernel.flatten();
    //    int kernel_size = kernel.shape(0);
    int dim_out = calculate_output_dim(image.shape(0), kernel_size, 0, stride, 1);
    ts::Tensor<float, 3> result(dim_out, dim_out, kernel.shape(1));

    for (int i = 0; i < dim_out; ++i) {
        for (int j = 0; j < dim_out; ++j) {
            ts::VectorF tile = get_flatten_tile(image, kernel_size, i*stride, j*stride);
            result(i, j) = ts::dot(ts::transpose(kernel), tile);
        }
    }
    return result;
}

auto pad(ts::MatrixF const & matrix, int pad_row, int pad_col)
{
    ts::MatrixF matrix_padded(matrix.shape(0) + 2 * pad_row, matrix.shape(1) + 2 * pad_col);
    for (int i = 0; i < matrix.shape(0); ++i) {
        for (int j = 0; j < matrix.shape(1); ++j) {
            matrix_padded(i + pad_row, j + pad_col) = matrix(i, j);
        }
    }
    return matrix_padded;
}

using namespace ts;

TEST_CASE("get_flatten_tile(Tensor<float, 3>, ...)")
{
    Tensor<float, 3> tensor = {{{0, 0}, {1, 0}, {2, 0}, {3, 0}},
                               {{0, 1}, {1, 1}, {2, 1}, {3, 1}},
                               {{0, 2}, {1, 2}, {2, 2}, {3, 2}},
                               {{0, 3}, {1, 3}, {2, 3}, {3, 3}}};
    {
        auto result = get_flatten_tile(tensor, 2, 0, 0);
        auto expected = VectorF{0, 0, 1, 0, 0, 1, 1, 1};
        REQUIRE(result == expected);
    }

    {
        auto result = get_flatten_tile(tensor, 2, 2, 2);
        auto expected = VectorF{2, 2, 3, 2, 2, 3, 3, 3};
        REQUIRE(result == expected);
    }
}

TEST_CASE("get_flatten_tile(MatrixF, ...)")
{
    MatrixF matrixF = {{10, 20, 30, 40},
                       {11, 21, 31, 41},
                       {12, 22, 32, 42},
                       {13, 23, 33, 43}};
    {
        auto result = get_flatten_tile(matrixF, 2, 0, 0);
        auto expected = VectorF{10, 20, 11, 21};
        REQUIRE(result == expected);
    }
    {
        auto result = get_flatten_tile(matrixF, 2, 0, 1);
        auto expected = VectorF{20, 30, 21, 31};
        REQUIRE(result == expected);
    }
    {
        auto result = get_flatten_tile(matrixF, 2, 1, 0);
        auto expected = VectorF{11, 21, 12, 22};
        REQUIRE(result == expected);
    }
    {
        auto result = get_flatten_tile(matrixF, 2, 2, 2);
        auto expected = VectorF{32, 42, 33, 43};
        REQUIRE(result == expected);
    }
    {
        auto result = get_flatten_tile(matrixF, 3, 0, 0);
        auto expected = VectorF{10, 20, 30, 11, 21, 31, 12, 22, 32};
        REQUIRE(result == expected);
    }
    {
        auto result = get_flatten_tile(matrixF, 3, 1, 1);
        auto expected = VectorF{21, 31, 41, 22, 32, 42, 23, 33, 43};
        REQUIRE(result == expected);
    }
}

TEST_CASE("conv2d")
{
    MatrixF matrix = {{10, 20, 30, 40},
                      {11, 21, 31, 41},
                      {12, 22, 32, 42},
                      {13, 23, 33, 43}};
    MatrixF kernel = {{2, 2},
                      {2, 2}};

    {
        auto result = conv2d(matrix, kernel, 1);
        MatrixF expected = {{124, 204, 284},
                            {132, 212, 292},
                            {140, 220, 300}};
        REQUIRE(result.data_size() == 9);
        REQUIRE(result == expected);
    }
    {
        auto result = conv2d(matrix, kernel, 2);
        MatrixF expected = {{124, 284},
                            {140, 300}};
        REQUIRE(result.data_size() == 4);
        REQUIRE(result == expected);
    }
}

TEST_CASE("pad")
{
    MatrixF matrix = {{1, 1},
                      {1, 1}};
    {
        auto result = pad(matrix, 1, 0);
        MatrixF expected = {{0, 0},
                            {1, 1},
                            {1, 1},
                            {0, 0}};
        REQUIRE(result == expected);
    }
    {
        auto result = pad(matrix, 0, 1);
        MatrixF expected = {{0, 1, 1, 0},
                            {0, 1, 1, 0}};
        REQUIRE(result == expected);
    }
    {
        auto result = pad(matrix, 2, 2);
        MatrixF expected = {
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {0, 0, 1, 1, 0, 0},
            {0, 0, 1, 1, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0}
        };
        REQUIRE(result == expected);
    }
}

TEST_CASE("conv2d gradient")
{
    // shape: (H, W, C_in)
    Tensor<float, 3> input =
        {{{1, -1}, {2, -2}, {3, -3}},
         {{4, -4}, {5, -5}, {6, -6}},
         {{7, -7}, {8, -8}, {9, -9}}};


    // shape: (k*k*C_in, c_out)
    Tensor<float, 2> kernel = {
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
        {2, 1},
        {-2, -1},
    };

    Tensor<float, 3> d_output = {
        {{1, -1}, {2, -2}},
        {{3, -3}, {4, -4}}
    };

    assert((kernel.shape() == std::array<int, 2>{8, 2}));

    int kernel_size = 2;
    int stride = 1;

    auto output = conv2d(input, kernel, kernel_size, stride);

    Tensor<float, 3> d_input(3, 3, 2);  // to compute
    Tensor<float, 2> d_kernel(8, 2);  // to compute

    for (int i = 0; i < d_output.shape(0); ++i) {
        for (int j = 0; j < d_output.shape(1); ++j) {
            ts::VectorF d_tile = d_output(i, j);
            ts::VectorF tile = get_flatten_tile(input, kernel_size, i*stride, j*stride);

            auto d_tile_kernel = ts::outer_product(tile, d_tile);
            auto d_tile_input = ts::dot(kernel, d_tile);

            add_flatten_tile(d_input, d_tile_input, kernel_size, i*stride, j*stride);
            ts::add_(d_kernel, d_tile_kernel);
        }
    }

    Tensor<float, 3> expected_output = {
        {{48, 24}, {64, 32}},
        {{96, 48}, {112, 56}}
    };

    Tensor<float, 3> expected_d_input =
        {{{1, -1}, {3, -3}, {2, -2}},
         {{4, -4}, {10, -10}, {6, -6}},
         {{3, -3}, {7, -7}, {4, -4}}};

    Tensor<float, 2> expected_d_kernel = {
        {37, -37},
        {-37, 37},
        {47, -47},
        {-47, 47},
        {67, -67},
        {-67, 67},
        {77, -77},
        {-77, 77},
    };

    REQUIRE(output == expected_output);
    REQUIRE(d_input == expected_d_input);
    REQUIRE(d_kernel == expected_d_kernel);
}
