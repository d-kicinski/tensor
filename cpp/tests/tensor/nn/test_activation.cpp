#include <catch2/catch.hpp>

#include <tensor/nn/activations.hpp>

using namespace ts;

TEST_CASE("ReLU<float, 2>")
{
    ReLU<float, 2> activation;

    MatrixF input = {{1, -1}, {-1, 1}};
    auto forward = activation(input);
    MatrixF expected_forward = {{1, 0}, {0, 1}};
    REQUIRE(forward == expected_forward);


    MatrixF d_output = {{2, 2}, {2, 2}};
    auto backward = activation.backward(d_output);
    MatrixF expected_backward = {{2, 0}, {0, 2}};
    REQUIRE(backward == expected_backward);
}

TEST_CASE("ReLU<float, 3>")
{
    ReLU<float, 3> activation;

    Tensor<float, 3> input = {{{1, -1}, {-1, 1}},
                              {{-1, 1}, {1, -1}}};
    auto forward = activation(input);
    Tensor<float, 3> expected_forward = {{{1, 0}, {0, 1}},
                                         {{0, 1}, {1, 0}}};
    REQUIRE(forward == expected_forward);


    Tensor<float, 3> d_output = {{{-2, 2}, {2, -2}},
                                 {{2, -2}, {-2, 2}}};
    auto backward = activation.backward(d_output);
    Tensor<float, 3> expected_backward = {{{-2, 0}, {0, -2}},
                                          {{0, -2}, {-2, 0}}};
    REQUIRE(backward == expected_backward);
}
